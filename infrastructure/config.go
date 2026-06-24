package infrastructure

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

type Config struct {
	Database  DatabaseConfig  `yaml:"database"`
	LLM       LLMConfig       `yaml:"llm"`
	Embedding EmbeddingConfig `yaml:"embedding"`
	WebSearch WebSearchConfig `yaml:"web_search"`
}

type WebSearchConfig struct {
	TavilyAPIKey string `yaml:"tavily_api_key"`
}

type DatabaseConfig struct {
	Host     string `yaml:"host"`
	Port     string `yaml:"port"`
	User     string `yaml:"user"`
	Password string `yaml:"password"`
	DBName   string `yaml:"dbname"`
	SSLMode  string `yaml:"sslmode"`
}

type LLMConfig struct {
	Provider string `yaml:"provider"`
	BaseUrl  string `yaml:"base_url"`
	APIKey   string `yaml:"api_key"`
	Model    string `yaml:"model"`
}

type EmbeddingConfig struct {
	Provider  string `yaml:"provider"`
	BaseUrl   string `yaml:"base_url"`
	APIKey    string `yaml:"api_key"`
	Model     string `yaml:"model"`
	Dimension int    `yaml:"dimension"`
}

func Load() (*Config, error) {
	data, err := os.ReadFile("config.yaml")
	if err != nil {
		return nil, fmt.Errorf("failed to read config.yaml: %w", err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config.yaml: %w", err)
	}
	return &cfg, nil
}

func InitDB(cfg DatabaseConfig) (*gorm.DB, error) {
	dsn := fmt.Sprintf("postgresql://%s:%s@%s:%s/%s?sslmode=%s",
		cfg.User, cfg.Password, cfg.Host, cfg.Port, cfg.DBName, cfg.SSLMode)
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}
	if err := db.Exec("CREATE EXTENSION IF NOT EXISTS vector").Error; err != nil {
		return nil, fmt.Errorf("failed to enable vector extension: %w", err)
	}

	return db, nil
}
