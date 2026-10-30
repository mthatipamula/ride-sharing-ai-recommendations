package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"golang.org/x/time/rate"
)

const (
	maxBodyBytes   = 64 << 10
	limiterIdleTTL = 10 * time.Minute
)

type Config struct {
	Upstream   *url.URL
	Timeout    time.Duration
	RiderRate  float64
	RiderBurst int
}

// Only these reach the service. /metrics, /docs and anything added later stay private until listed here.
var routes = map[string]string{
	"POST /v1/recommendations":       "recommendations",
	"POST /v1/ride-options":          "ride_options",
	"POST /v1/events/ride-completed": "ride_completed",
	"GET /v1/riders/{id}":            "rider",
	"GET /v1/places":                 "places",
	"GET /health/ready":              "ready",
}

type Gateway struct {
	mux      *http.ServeMux
	proxy    *httputil.ReverseProxy
	cfg      Config
	limiters *limiters
	breaker  *breaker
	log      *slog.Logger

	requests     *prometheus.CounterVec
	duration     *prometheus.HistogramVec
	rateLimited  prometheus.Counter
	shortCircuit prometheus.Counter
}

func New(cfg Config, reg prometheus.Registerer, log *slog.Logger) *Gateway {
	g := &Gateway{
		mux: http.NewServeMux(), cfg: cfg, log: log,
		limiters: newLimiters(rate.Limit(cfg.RiderRate), cfg.RiderBurst),
		breaker:  newBreaker(5, 5*time.Second),
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{Name: "gateway_requests_total",
			Help: "Requests by route and status"}, []string{"route", "status"}),
		duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "gateway_request_duration_seconds",
			Help:    "Gateway latency including upstream",
			Buckets: []float64{.002, .005, .01, .02, .03, .05, .075, .1, .15, .25, .5, 1}}, []string{"route"}),
		rateLimited: prometheus.NewCounter(prometheus.CounterOpts{Name: "gateway_rate_limited_total",
			Help: "Requests rejected by the per-rider limit"}),
		shortCircuit: prometheus.NewCounter(prometheus.CounterOpts{Name: "gateway_breaker_rejected_total",
			Help: "Requests rejected while the breaker was open"}),
	}
	reg.MustRegister(g.requests, g.duration, g.rateLimited, g.shortCircuit,
		prometheus.NewGaugeFunc(prometheus.GaugeOpts{Name: "gateway_breaker_open",
			Help: "1 while the upstream circuit breaker is open"}, func() float64 {
			if g.breaker.isOpen() {
				return 1
			}
			return 0
		}))

	g.proxy = &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(cfg.Upstream)
			r.SetXForwarded()
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			status, msg := http.StatusBadGateway, "recommendations unavailable"
			if errors.Is(err, context.DeadlineExceeded) {
				status, msg = http.StatusGatewayTimeout, "recommendations timed out"
			}
			writeError(w, status, msg)
		},
	}
	for pattern, name := range routes {
		g.mux.Handle(pattern, g.route(name))
	}
	return g
}

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) { g.mux.ServeHTTP(w, r) }

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (s *statusWriter) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

func (g *Gateway) route(name string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		id := r.Header.Get("X-Request-Id")
		if id == "" {
			id = newID()
			r.Header.Set("X-Request-Id", id)
		}
		w.Header().Set("X-Request-Id", id)
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		g.handle(sw, r, name)
		elapsed := time.Since(start)
		g.requests.WithLabelValues(name, strconv.Itoa(sw.status)).Inc()
		g.duration.WithLabelValues(name).Observe(elapsed.Seconds())
		g.log.Info("request", "id", id, "route", name, "status", sw.status, "ms", elapsed.Milliseconds())
	})
}

func (g *Gateway) handle(w *statusWriter, r *http.Request, name string) {
	key, err := limitKey(r)
	if err != nil {
		writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
		return
	}
	if !g.limiters.allow(key) {
		g.rateLimited.Inc()
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusTooManyRequests, "too many requests")
		return
	}
	guarded := name != "ready" // a readiness probe must report the upstream's state, not change the breaker
	if guarded && !g.breaker.allow() {
		g.shortCircuit.Inc()
		w.Header().Set("Retry-After", "5")
		writeError(w, http.StatusServiceUnavailable, "recommendations temporarily unavailable")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), g.cfg.Timeout)
	defer cancel()
	g.proxy.ServeHTTP(w, r.WithContext(ctx))
	if guarded {
		g.breaker.record(w.status < 500)
	}
}

// limitKey buffers the body (bounded) and limits by rider when the request names one, otherwise by client IP.
func limitKey(r *http.Request) (string, error) {
	if r.Body != nil && r.Method == http.MethodPost {
		body, err := io.ReadAll(http.MaxBytesReader(nil, r.Body, maxBodyBytes))
		if err != nil {
			return "", err
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		var parsed struct {
			RiderID *int64 `json:"rider_id"`
		}
		if json.Unmarshal(body, &parsed) == nil && parsed.RiderID != nil {
			return "rider:" + strconv.FormatInt(*parsed.RiderID, 10), nil
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	return "ip:" + host, nil
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"detail": msg})
}

func newID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// limiters is one token bucket per rider, evicted when idle so memory tracks active riders.
type limiters struct {
	mu     sync.Mutex
	r      rate.Limit
	burst  int
	byKey  map[string]*entry
	nextGC time.Time
}

type entry struct {
	limiter *rate.Limiter
	seen    time.Time
}

func newLimiters(r rate.Limit, burst int) *limiters {
	return &limiters{r: r, burst: burst, byKey: map[string]*entry{}, nextGC: time.Now().Add(limiterIdleTTL)}
}

func (l *limiters) allow(key string) bool {
	now := time.Now()
	l.mu.Lock()
	defer l.mu.Unlock()
	if now.After(l.nextGC) {
		for k, e := range l.byKey {
			if now.Sub(e.seen) > limiterIdleTTL {
				delete(l.byKey, k)
			}
		}
		l.nextGC = now.Add(limiterIdleTTL)
	}
	e, ok := l.byKey[key]
	if !ok {
		e = &entry{limiter: rate.NewLimiter(l.r, l.burst)}
		l.byKey[key] = e
	}
	e.seen = now
	return e.limiter.AllowN(now, 1)
}

// breaker opens after N consecutive upstream failures, then lets one trial request through after the cooldown.
type breaker struct {
	mu        sync.Mutex
	threshold int
	cooldown  time.Duration
	failures  int
	openUntil time.Time
	trial     bool
}

func newBreaker(threshold int, cooldown time.Duration) *breaker {
	return &breaker{threshold: threshold, cooldown: cooldown}
}

func (b *breaker) isOpen() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.failures >= b.threshold
}

func (b *breaker) allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.failures < b.threshold {
		return true
	}
	if time.Now().Before(b.openUntil) || b.trial {
		return false
	}
	b.trial = true
	return true
}

func (b *breaker) record(ok bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.trial = false
	if ok {
		b.failures = 0
		return
	}
	b.failures++
	if b.failures >= b.threshold {
		b.openUntil = time.Now().Add(b.cooldown)
	}
}
