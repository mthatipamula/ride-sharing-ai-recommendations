// API gateway for the rider app: the only public entry point to the recommendation service.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(key), 64); err == nil {
		return v
	}
	return fallback
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	upstream, err := url.Parse(env("UPSTREAM_URL", "http://localhost:8080"))
	if err != nil {
		logger.Error("bad UPSTREAM_URL", "error", err)
		os.Exit(1)
	}

	reg := prometheus.NewRegistry()
	gw := New(Config{
		Upstream:   upstream,
		Timeout:    time.Duration(envFloat("UPSTREAM_TIMEOUT_MS", 250)) * time.Millisecond,
		RiderRate:  envFloat("RIDER_RATE_PER_SEC", 20),
		RiderBurst: int(envFloat("RIDER_BURST", 40)),
	}, reg, logger)

	public := &http.Server{Addr: env("LISTEN_ADDR", ":8090"), Handler: gw, ReadHeaderTimeout: 5 * time.Second}
	// metrics stay on their own port, so they are never reachable through the public route table
	admin := &http.Server{Addr: env("ADMIN_ADDR", ":9090"), Handler: promhttp.HandlerFor(reg, promhttp.HandlerOpts{}),
		ReadHeaderTimeout: 5 * time.Second}

	for _, s := range []*http.Server{public, admin} {
		go func() {
			logger.Info("listening", "addr", s.Addr)
			if err := s.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				logger.Error("server failed", "addr", s.Addr, "error", err)
				os.Exit(1)
			}
		}()
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = public.Shutdown(shutdown)
	_ = admin.Shutdown(shutdown)
}
