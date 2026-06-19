package internal

import (
	"fmt"
	"strings"
)

func SchemaName(lesson string) (string, error) {
	if lesson == "" {
		return "", fmt.Errorf("lesson name is empty")
	}
	lower := strings.ToLower(lesson)
	for _, r := range lower {
		ok := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_'
		if !ok {
			return "", fmt.Errorf("invalid lesson name %q: only [a-z0-9_] allowed", lesson)
		}
	}
	return "l_" + lower, nil
}
