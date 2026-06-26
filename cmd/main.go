package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"gorm.io/gorm"

	"rag/internal"

	// 注册各章节的 lesson(init() 副作用)。
	_ "rag/internal/ch01"
	_ "rag/internal/ch02"
	_ "rag/internal/ch03"
	_ "rag/internal/ch04"
	_ "rag/internal/ch05"
	_ "rag/internal/ch06"
	_ "rag/internal/ch07"
	_ "rag/internal/ch08"
	_ "rag/internal/ch08neo4j"
	_ "rag/internal/ch09"
	_ "rag/internal/ch10"

	"rag/infrastructure"
)

func main() {
	logger, logFile, err := infrastructure.InitLogger(chapterName(os.Args))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer logFile.Close()

	cfg, err := infrastructure.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	gormDB, err := infrastructure.InitDB(cfg.Database)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	sqlDB, err := gormDB.DB()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer sqlDB.Close()

	llm, err := infrastructure.NewLLM(cfg.LLM)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	emb, err := infrastructure.NewEmbedder(cfg.Embedding)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	neo4jDrv, err := infrastructure.NewNeo4j(cfg.Neo4j)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if neo4jDrv != nil {
		defer neo4jDrv.Close(context.Background())
	}

	deps := &internal.Deps{
		DB:       gormDB,
		Neo4j:    neo4jDrv,
		LLM:      llm,
		Embedder: emb,
		Cfg:      cfg,
		Logger:   logger,
	}

	root := &cobra.Command{
		Use:   "rag",
		Short: "rag lesson runner",
		Long:  "Run any registered lesson with `rag <lesson-name> [flags]`.",
	}
	root.AddCommand(migrateCmd(gormDB, deps))
	for _, l := range internal.All() {
		root.AddCommand(lessonCmd(l, deps))
	}
	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

// lessonCmd 把 Lesson 包成 cobra 子命令：start transaction → set search_path → call Run.
func lessonCmd(l internal.Lesson, base *internal.Deps) *cobra.Command {
	return &cobra.Command{
		Use:                l.Name,
		Short:              l.Description,
		DisableFlagParsing: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			schema, err := infrastructure.SchemaName(l.Name)
			if err != nil {
				return err
			}
			ctx := cmd.Context()
			if ctx == nil {
				ctx = context.Background()
			}
			return base.DB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
				if err := tx.Exec("SET LOCAL search_path TO " + schema + ", public").Error; err != nil {
					return fmt.Errorf("set search_path: %w", err)
				}
				deps := *base
				deps.DB = tx
				return l.Run(ctx, deps, args)
			})
		},
	}
}

func migrateCmd(gormDB *gorm.DB, deps *internal.Deps) *cobra.Command {
	var all bool
	cmd := &cobra.Command{
		Use:   "migrate [lesson-name]",
		Short: "Run DDL for a lesson (or --all)",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			targets := internal.All()
			if !all {
				if len(args) == 0 {
					return fmt.Errorf("specify a lesson name or pass --all")
				}
				name := args[0]
				filtered := make([]internal.Lesson, 0, 1)
				for _, l := range targets {
					if l.Name == name {
						filtered = append(filtered, l)
					}
				}
				if len(filtered) == 0 {
					return fmt.Errorf("unknown lesson: %s", name)
				}
				targets = filtered
			}

			for _, l := range targets {
				if l.Migrate == nil {
					slog.Info(fmt.Sprintf("- %s: no Migrate func, skipping", l.Name))
					continue
				}
				schema, err := infrastructure.SchemaName(l.Name)
				if err != nil {
					return err
				}
				if err := gormDB.Exec("CREATE SCHEMA IF NOT EXISTS " + schema).Error; err != nil {
					return fmt.Errorf("create schema %s: %w", schema, err)
				}
				ctx := cmd.Context()
				if ctx == nil {
					ctx = context.Background()
				}
				migDeps := *deps
				err = gormDB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
					if err := tx.Exec("SET LOCAL search_path TO " + schema + ", public").Error; err != nil {
						return fmt.Errorf("set search_path: %w", err)
					}
					migDeps.DB = tx
					return l.Migrate(ctx, migDeps)
				})
				if err != nil {
					return fmt.Errorf("migrate %s: %w", l.Name, err)
				}
				slog.Info(fmt.Sprintf("✓ %s -> schema %s", l.Name, schema))
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&all, "all", false, "migrate every registered lesson")
	return cmd
}

// chapterName 从 os.Args 提取日志文件名用的 chapter 名。
// `rag graph` → "graph";`rag migrate graph` → "graph";`rag migrate --all` → "migrate";否则 "rag"。
func chapterName(argv []string) string {
	if len(argv) < 2 {
		return "rag"
	}
	args := argv[1:]
	if args[0] == "migrate" {
		if len(args) >= 2 && !strings.HasPrefix(args[1], "--") {
			return args[1]
		}
		return "migrate"
	}
	return args[0]
}
