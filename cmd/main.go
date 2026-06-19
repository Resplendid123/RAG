// cmd/main.go
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/spf13/cobra"
	"gorm.io/gorm"

	ragpkg "rag/rag"

	"rag/internal"
)

func main() {
	cfg, err := internal.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	gormDB, err := internal.InitDB(cfg.Database)
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

	llm, err := internal.NewLLM(cfg.LLM)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	emb, err := internal.NewEmbedder(cfg.Embedding)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	deps := &ragpkg.Deps{
		DB:       gormDB,
		LLM:      llm,
		Embedder: emb,
		Cfg:      cfg,
		Logger:   slog.Default(),
	}

	root := &cobra.Command{
		Use:   "rag",
		Short: "rag lesson runner",
		Long:  "Run any registered lesson with `rag <lesson-name> [flags]`.",
	}
	root.AddCommand(migrateCmd(gormDB))
	for _, l := range ragpkg.All() {
		root.AddCommand(lessonCmd(l, deps))
	}
	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

// lessonCmd 把 Lesson 包成 cobra 子命令：start transaction → set search_path → call Run.
func lessonCmd(l ragpkg.Lesson, base *ragpkg.Deps) *cobra.Command {
	return &cobra.Command{
		Use:                l.Name,
		Short:              l.Description,
		DisableFlagParsing: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			schema, err := internal.SchemaName(l.Name)
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

func migrateCmd(gormDB *gorm.DB) *cobra.Command {
	var all bool
	cmd := &cobra.Command{
		Use:   "migrate [lesson-name]",
		Short: "Run DDL for a lesson (or --all)",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			targets := ragpkg.All()
			if !all {
				if len(args) == 0 {
					return fmt.Errorf("specify a lesson name or pass --all")
				}
				name := args[0]
				filtered := make([]ragpkg.Lesson, 0, 1)
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
					fmt.Printf("- %s: no Migrate func, skipping\n", l.Name)
					continue
				}
				schema, err := internal.SchemaName(l.Name)
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
				err = gormDB.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
					if err := tx.Exec("SET LOCAL search_path TO " + schema + ", public").Error; err != nil {
						return fmt.Errorf("set search_path: %w", err)
					}
					return l.Migrate(ctx, tx)
				})
				if err != nil {
					return fmt.Errorf("migrate %s: %w", l.Name, err)
				}
				fmt.Printf("✓ %s -> schema %s\n", l.Name, schema)
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&all, "all", false, "migrate every registered lesson")
	return cmd
}
