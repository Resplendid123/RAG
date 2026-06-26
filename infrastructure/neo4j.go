package infrastructure

import (
	"fmt"

	"github.com/neo4j/neo4j-go-driver/v6/neo4j"
)

// NewNeo4j 起 bolt driver;URI 为空时返回 nil。延迟到首次 Session.Run 才连,启动期不阻塞其他 chapter。
func NewNeo4j(cfg Neo4jConfig) (neo4j.Driver, error) {
	if cfg.URI == "" {
		return nil, nil
	}
	driver, err := neo4j.NewDriver(cfg.URI, neo4j.BasicAuth(cfg.Username, cfg.Password, ""))
	if err != nil {
		return nil, fmt.Errorf("neo4j driver: %w", err)
	}
	return driver, nil
}
