package main

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

func setup(t *testing.T, upstream http.HandlerFunc, cfg Config) (*Gateway, *httptest.Server) {
	t.Helper()
	backend := httptest.NewServer(upstream)
	t.Cleanup(backend.Close)
	u, _ := url.Parse(backend.URL)
	cfg.Upstream = u
	if cfg.Timeout == 0 {
		cfg.Timeout = time.Second
	}
	if cfg.RiderRate == 0 {
		cfg.RiderRate, cfg.RiderBurst = 1000, 1000
	}
	g := New(cfg, prometheus.NewRegistry(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	front := httptest.NewServer(g)
	t.Cleanup(front.Close)
	return g, front
}

func post(t *testing.T, base, path, body string) *http.Response {
	t.Helper()
	r, err := http.Post(base+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	_, _ = io.Copy(io.Discard, r.Body)
	return r
}

func ok(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }

func TestProxiesAllowListedRouteAndKeepsBody(t *testing.T) {
	var got string
	_, front := setup(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = string(b)
	}, Config{})
	r := post(t, front.URL, "/v1/recommendations", `{"rider_id":1}`)
	if r.StatusCode != 200 || got != `{"rider_id":1}` {
		t.Fatalf("status %d body %q", r.StatusCode, got)
	}
	if r.Header.Get("X-Request-Id") == "" {
		t.Fatal("missing request id")
	}
}

func TestPrivatePathsAreNotExposed(t *testing.T) {
	_, front := setup(t, ok, Config{})
	for _, path := range []string{"/metrics", "/metrics/", "/docs", "/openapi.json"} {
		r, err := http.Get(front.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		r.Body.Close()
		if r.StatusCode != http.StatusNotFound {
			t.Errorf("%s: got %d", path, r.StatusCode)
		}
	}
}

func TestRateLimitIsPerRider(t *testing.T) {
	_, front := setup(t, ok, Config{RiderRate: 0.001, RiderBurst: 2})
	for i := 0; i < 2; i++ {
		if s := post(t, front.URL, "/v1/recommendations", `{"rider_id":1}`).StatusCode; s != 200 {
			t.Fatalf("request %d: %d", i, s)
		}
	}
	if s := post(t, front.URL, "/v1/recommendations", `{"rider_id":1}`).StatusCode; s != 429 {
		t.Fatalf("over limit: %d", s)
	}
	if s := post(t, front.URL, "/v1/recommendations", `{"rider_id":2}`).StatusCode; s != 200 {
		t.Fatalf("other rider: %d", s)
	}
}

func TestDeadlineReturns504(t *testing.T) {
	_, front := setup(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(time.Second):
		case <-r.Context().Done():
		}
	}, Config{Timeout: 30 * time.Millisecond})
	if s := post(t, front.URL, "/v1/recommendations", `{"rider_id":1}`).StatusCode; s != 504 {
		t.Fatalf("got %d", s)
	}
}

func TestBreakerOpensThenRecovers(t *testing.T) {
	failing := true
	calls := 0
	g, front := setup(t, func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if failing {
			w.WriteHeader(http.StatusInternalServerError)
		}
	}, Config{})
	g.breaker.cooldown = 40 * time.Millisecond
	for i := 0; i < 5; i++ {
		post(t, front.URL, "/v1/ride-options", `{"rider_id":1}`)
	}
	if s := post(t, front.URL, "/v1/ride-options", `{"rider_id":1}`).StatusCode; s != 503 || calls != 5 {
		t.Fatalf("open breaker: status %d, upstream calls %d", s, calls)
	}
	failing = false
	time.Sleep(60 * time.Millisecond)
	if s := post(t, front.URL, "/v1/ride-options", `{"rider_id":1}`).StatusCode; s != 200 {
		t.Fatalf("trial: %d", s)
	}
	if g.breaker.isOpen() {
		t.Fatal("breaker should have closed")
	}
}

func TestOversizedBodyRejected(t *testing.T) {
	_, front := setup(t, ok, Config{})
	big := `{"rider_id":1,"x":"` + strings.Repeat("a", maxBodyBytes) + `"}`
	if s := post(t, front.URL, "/v1/recommendations", big).StatusCode; s != 413 {
		t.Fatalf("got %d", s)
	}
}

func TestReadinessProbeDoesNotTripBreaker(t *testing.T) {
	g, front := setup(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}, Config{})
	for i := 0; i < 8; i++ {
		r, err := http.Get(front.URL + "/health/ready")
		if err != nil {
			t.Fatal(err)
		}
		r.Body.Close()
		if r.StatusCode != http.StatusServiceUnavailable {
			t.Fatalf("got %d", r.StatusCode)
		}
	}
	if g.breaker.isOpen() {
		t.Fatal("probe opened the breaker")
	}
}
