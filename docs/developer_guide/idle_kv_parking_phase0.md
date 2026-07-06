# Idle KV Parking — Phase 0 Baseline (Runbook + 코드 조사 결과)

> 목표: **새 코드를 짜기 전에**, sglang의 기존 hierarchical cache / decode-offload 경로가
> agentic multi-turn 워크로드에서 "재계산(recompute) vs fetch(reuse)"를 **이미 어디까지 커버하는지**
> 1P1D로 측정한다. 이 baseline이 이후 Phase 1(유휴 P GPU parking)의 이득을 재는 기준선이 된다.
>
> 측정 하니스는 `experiments` 저장소에 있음 (동일 브랜치 `claude/youthful-knuth-det52g`):
> `scripts/sglang/{start_1P_1D,stop_1P_1D,run_phase0_baseline}.sh`,
> `benchmark/{phase0_metrics_scraper,phase0_analyze}.py`.

---

## 1. 코드 조사 결과 (Phase 0의 "판단" 항목)

### 1.1 기존 재사용 메커니즘의 위치

multi-turn에서 turn N+1은 turn N과 긴 prefix를 공유한다. 이 prefix를 재계산하지 않고 재사용하는 경로:

| 계층 | 활성 방법 | 재사용 주체 |
|---|---|---|
| **L1 (GPU radix)** | 기본값(`--disable-radix-cache`로 끔) | Prefill 노드가 GPU에 남은 prefix를 hit |
| **L2 (CPU DRAM host pool)** | `--enable-hierarchical-cache` | Prefill 노드가 host로 offload한 prefix를 load-back |
| **L3 (storage)** | `--enable-hierarchical-cache --hicache-storage-backend <file/mooncake/...>` | Prefill 노드가 storage에서 prefetch |
| **Decode offload** | `--disaggregation-decode-enable-offload-kvcache` (decode 측) | Decode KV를 host/storage로 내림 |

→ multi-turn "fetch가 recompute를 대체"하는 핵심은 **Prefill 노드의 L1/L2/L3 prefix 재사용**이다.

### 1.2 host-only 완화 필요성 — 판단

- **Prefill 쪽은 이미 host-only 가능**: `--enable-hierarchical-cache`만 켜면 storage backend 없이도 **L1+L2(GPU+CPU DRAM)** 로 동작한다. 즉 설계 문서가 말한 "Tier3 CPU DRAM fetch"의 상당 부분은 **오늘 그대로 측정 가능**하며 별도 코드 완화가 필요 없다.
- **Decode offload만 storage backend를 강제**한다:
  `python/sglang/srt/server_args.py:4346-4354`
  ```python
  if self.disaggregation_decode_enable_offload_kvcache:
      if self.disaggregation_mode != "decode":
          raise ValueError(...only supported for decode side.)
      if self.hicache_storage_backend is None:
          raise ValueError(...only supported when hicache-storage-backend is provided.)
  ```
  따라서 decode KV를 **CPU DRAM만으로(host-only)** offload하는 건 현재 **불가**.
- **결론**: Phase 0에서는 완화가 **불필요**. Prefill L2 host-only 재사용으로 baseline을 충분히 측정할 수 있다. decode offload host-only 완화는 (필요하다면) Phase 1+ 후보이며, 지금은 `hicache_file` 모드로 storage 포함 측정만 확보한다.

### 1.3 측정에 쓸 메트릭 (`--enable-metrics`)

`python/sglang/srt/observability/metrics_collector.py` 노출:

| 메트릭 | 의미 |
|---|---|
| `sglang:prompt_tokens_total` | 전체 prefill 입력 토큰 |
| `sglang:cached_tokens_total` | 그 중 캐시 재사용(=**fetch/reuse**) 토큰 |
| `prompt - cached` | 실제 **재계산(recompute)** 토큰 |
| `sglang:prefetched_tokens_total` | **L3(storage)** 에서 당겨온 토큰 |
| `sglang:cache_hit_rate` | prefix cache hit rate (gauge) |
| `sglang:hicache_host_used_tokens` / `_total_tokens` | **L2 host pool** 사용/용량 |
| `sglang:time_to_first_token_seconds` (_sum/_count) | TTFT |
| `sglang:num_used_tokens`, `sglang:token_usage` | GPU KV 점유 |

핵심 파생 지표: **reuse_ratio = cached / prompt** (fetch 비중), **recompute_ratio = 1 − reuse_ratio**.

---

## 2. 실험 설계 (config matrix)

1P1D, BFCL v3 multi-turn base, 순차(C=1). `CACHE_MODE`만 바꿔 5종 비교:

| CACHE_MODE | Prefill 인자 | 의미 |
|---|---|---|
| `none` | `--disable-radix-cache` | 매 턴 full re-prefill = **recompute 바닥값** |
| `radix` | (기본) | GPU L1만 |
| `hicache_host` | `--enable-hierarchical-cache --hicache-ratio R` | L1+L2 (**host-only**, storage 없음) |
| `hicache_file` | `+ --hicache-storage-backend file` | L1+L2+L3 (현 실험 기본) |
| `hicache_file_decode_offload` | `+` decode `--disaggregation-decode-enable-offload-kvcache` | decode KV까지 offload |

### 가설
`none → radix → hicache_host → hicache_file`로 갈수록 **reuse_ratio↑, avg TTFT↓, per-turn TTFT 성장 완만**해야 한다.
= "fetch가 recompute를 대체해 이득"이라는 아이디어 전제의 실측 확인. break-even 및 이득 상한이 Phase 1의 목표선이 된다.

---

## 3. 실행 방법 (server17)

```bash
cd ~/experiments
conda activate sglang

# 전체 스윕 (5개 모드 자동)
./scripts/sglang/run_phase0_baseline.sh

# 일부 모드만
MODES="none radix hicache_file" ./scripts/sglang/run_phase0_baseline.sh

# 단일 모드 수동
CACHE_MODE=hicache_file ./scripts/sglang/start_1P_1D.sh
CONFIG=phase0_hicache_file python benchmark/sglang_BFCL_v3_multi_turn_base.py
./scripts/sglang/stop_1P_1D.sh
```

산출물 (`results/phase0/`):
- `metrics_<mode>_{before,after,delta}.json` — 재계산 vs fetch 원자료/요약
- `bfcl_multiturn_results_phase0_<mode>.json` — TTFT/TPOT/throughput
- `phase0_summary.{md,csv}` — 모드 비교 테이블 + per-turn TTFT

> RAM 125GB 제약: `HICACHE_RATIO`(기본 2.0)를 과하게 올리지 말 것. 1P1D는 GPU 2장만 써서
> 2P2D(hicache-ratio 1.2)보다 여유가 있으나, host pool이 커지면 OOM 위험.

---

## 4. Phase 0 완료 기준 (Definition of Done) — ✅ 완료

- [x] 모드별 결과 수집 (`phase0_summary.md`)
- [x] reuse_ratio ↔ avg TTFT 상관 확인
- [x] L2/L3 기여 분해
- [x] fetch-vs-recompute를 실측으로 확인
- [x] Phase 1이 이겨야 할 baseline 확정

---

## 5. Phase 0 실측 결론 (2×A6000, 1P1D, BFCL v3 multi-turn 200개)

세 regime을 측정해 결론에 도달했다.

| Regime | 설정 | radix reuse | hicache reuse | radix TTFT | hicache TTFT | 해석 |
|---|---|---|---|---|---|---|
| **A. baseline** | C=1, 大 pool | 0.745 | 0.745 | 0.61s | 0.64s | GPU radix가 모든 prefix 확보 → 계층 캐시 idle (오히려 오버헤드) |
| **B. raw pressure** | pool 30k, C=16 | 0.06 | 0.35 (file, L3=924k) | 10.3s | 8.9s | 계층 활성화되나 **과부하**(TTFT는 queueing 지배) |
| **C. pressure + idle** ⭐ | pool 40k, C=8, TOOL_DELAY=3s | **0.257** | **0.743** | **2.40s** | **1.45s** | **clean: fetch가 recompute를 TTFT로 이김** |

### 핵심 결론
- **Regime C가 프로젝트 시나리오(tool-call 유휴 중 eviction)를 정확히 모델링한다.** 세션이 유휴(think-time)인 동안 다른 세션이 그 prefix를 evict → 다음 턴에:
  - `radix`(GPU only): 재계산 → reuse 0.257, TTFT 2.40s.
  - `hicache_host`(+L2 CPU DRAM): host에서 fetch → reuse 0.743 회복, **TTFT 1.45s (−40%)**, throughput +15%.
- 유휴가 부하를 분산해 queueing이 없으므로, **"fetch < recompute"가 TTFT로 깨끗하게 드러난다.** per-turn: radix는 유휴 후 매 턴 ~2.2s로 튐(재계산 반복), hicache는 ~1.2s 평탄(fetch).
- Regime C에서 `L3_prefetched=0` → pool 40k / host 80k라 **L2(host DRAM)만으로 충분**했고 disk L3까지 가지 않음.

### Phase 1(유휴 P GPU parking)에 대한 함의 — 중요
- 이 스케일에선 **L2(로컬 host DRAM)만으로 이미 fetch<recompute 이득을 낸다.** 따라서 Phase 1(원격 유휴 P GPU parking)의 가치는 "L2보다 빠른 fetch"가 아니라 다음에 있다:
  1. **용량**: 단일 노드 host RAM이 부족해 L2가 넘치면 → 느린 disk L3로 spill. 원격 유휴 P GPU HBM은 L2와 L3-disk 사이의 **크고 빠른 tier**를 제공.
  2. **자원 격리**: decode 노드의 host RAM(실험 환경에서 빠듯)을 소모하지 않고 원격 GPU로 offload.
- → **Phase 1 검증은 "host L2가 넘치는" 더 강한 압박(긴 대화 / 많은 세션이 host pool 초과)에서 해야 한다.** 그때 `radix→L2→L3(disk)` 경로가 느려지고, 원격 GPU parking이 disk L3를 이기는 것을 보여야 한다. Regime C의 (radix 2.40s vs L2 1.45s)가 Phase 1이 최소한 유지해야 할 기준선이다.

측정 하니스: `experiments/scripts/sglang/run_phase0_pressure.sh` (knob: `PREFILL_MAX_TOTAL_TOKENS`, `CONCURRENCY`, `TOOL_DELAY`), 결과는 `results/phase0_pressure/p<pool>_c<C>_d<delay>/`.

관련: 설계 [`idle_kv_parking_design.md`](./idle_kv_parking_design.md) · 배경/현실성 [`idle_kv_parking_related_work.md`](./idle_kv_parking_related_work.md)
