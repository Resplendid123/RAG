package ch08neo4j

import (
	"context"
	"fmt"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
)

// readSession 起一个 read session;v6 NewSession 不返回 error,DatabaseName 走 SessionConfig。
func readSession(ctx context.Context, drv neo4j.Driver) (neo4j.Session, error) {
	if drv == nil {
		return nil, fmt.Errorf("neo4j driver is nil")
	}
	return drv.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead, DatabaseName: "neo4j"}), nil
}

// writeSession 同上,write 模式。
func writeSession(ctx context.Context, drv neo4j.Driver) (neo4j.Session, error) {
	if drv == nil {
		return nil, fmt.Errorf("neo4j driver is nil")
	}
	return drv.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite, DatabaseName: "neo4j"}), nil
}
