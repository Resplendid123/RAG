package ch03

import (
	"sort"
)

// RRF Reciprocal Rank Fusion:不看具体分数量纲,只看"在各自列表里排第几"。
// score(d) = sum( 1 / (k + rank_i(d))  for i in each retriever )
// k 常取 60(Microsoft Bing 推荐折中点);k 越小排第一越主导,k 越大家越平等。
// 同一 ChunkID 在多路都出现就累加;Content 优先取第一次出现的(dense 优先)。
func RRF(lists [][]Hit, k int) []Hit {
	if k <= 0 {
		k = 60
	}
	scores := make(map[int64]float64)
	byID := make(map[int64]Hit)
	for _, list := range lists {
		for _, h := range list {
			scores[h.ChunkID] += 1.0 / float64(k+h.Rank)
			if _, ok := byID[h.ChunkID]; !ok {
				byID[h.ChunkID] = h
			}
		}
	}
	out := make([]Hit, 0, len(byID))
	for id, h := range byID {
		h.Rank = 0 // 原始排名
		out = append(out, h)
		_ = id
	}
	sort.Slice(out, func(i, j int) bool {
		if scores[out[i].ChunkID] != scores[out[j].ChunkID] {
			return scores[out[i].ChunkID] > scores[out[j].ChunkID]
		}
		return out[i].ChunkID < out[j].ChunkID
	})
	for i := range out {
		out[i].Rank = i
	}
	return out
}
