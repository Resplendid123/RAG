package infrastructure

import (
	"fmt"
	"strings"
)

func SchemaName(lesson string) (string, error) {
	if lesson == "" {
		return "", fmt.Errorf("lesson name is empty")
	}
	lower := strings.ToLower(lesson)
	return "l_" + lower, nil
}
