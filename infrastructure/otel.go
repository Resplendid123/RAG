package infrastructure

import (
	"context"
	"fmt"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

// tracerName 是 otel.Tracer() 注册名,跟 serviceName 区分(后者是 resource attribute)。
// Langfuse / Jaeger / 其它后端按 tracer name 区分来源。
const tracerName = "rag"

// InitTracer 配全局 TracerProvider,返回 shutdown 函数(defer 调,flush span)。
//   - OTEL_EXPORTER_OTLP_ENDPOINT 设了 → OTLP HTTP exporter(Langfuse / 通用后端);
//     Authorization header 走 OTEL_EXPORTER_OTLP_HEADERS(逗号分隔 key=value),跟 OTel 标准环境变量对齐。
//   - 没设 → stdout exporter(本地 demo 可看,不依赖外部服务)。
func InitTracer(ctx context.Context, serviceName string) (func(context.Context) error, error) {
	var exporter sdktrace.SpanExporter
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint != "" {
		opts := []otlptracehttp.Option{otlptracehttp.WithEndpointURL(endpoint)}
		if headers := os.Getenv("OTEL_EXPORTER_OTLP_HEADERS"); headers != "" {
			opts = append(opts, otlptracehttp.WithHeaders(parseHeaders(headers)))
		}
		client := otlptracehttp.NewClient(opts...)
		exp, err := otlptrace.New(ctx, client)
		if err != nil {
			return nil, fmt.Errorf("otlp http exporter: %w", err)
		}
		exporter = exp
	} else {
		exp, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
		if err != nil {
			return nil, fmt.Errorf("stdout exporter: %w", err)
		}
		exporter = exp
	}

	res, err := resource.New(ctx, resource.WithAttributes(
		semconv.ServiceName(serviceName),
	))
	if err != nil {
		return nil, fmt.Errorf("resource: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)
	return tp.Shutdown, nil
}

// Tracer 返回全局 Tracer;包内任何位置都能拿到同一个。
func Tracer() trace.Tracer { return otel.Tracer(tracerName) }

// parseHeaders 解析 "k1=v1,k2=v2" 形式的 OTEL_EXPORTER_OTLP_HEADERS。
func parseHeaders(s string) map[string]string {
	out := map[string]string{}
	cur, key := "", ""
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '=':
			key = cur
			cur = ""
		case ',':
			if key != "" {
				out[key] = cur
			}
			key, cur = "", ""
		default:
			cur += string(s[i])
		}
	}
	if key != "" {
		out[key] = cur
	}
	return out
}
