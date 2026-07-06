# Idle P-node KV Parking 설계 문서

> **상태**: Draft / 설계 단계
> **대상 워크로드**: Agentic multi-turn (예: BFCL v3 multi-turn) — tool-call 유휴시간이 많은 PD disaggregation 환경
> **목표**: Decode(D) node가 요청 종료 후 KV를 free하는 대신, tool-call 유휴시간 동안 유휴 Prefill(P) node의 GPU radix cache로 park하여 다음 턴의 재-prefill을 스킵한다. GPU 용량 초과 시 P의 CPU DRAM(Tier3)으로 강등한다.

---

## 1. 배경 및 문제 정의

### 1.1 현재 PD disaggregation 흐름

```
Turn N:
  [P prefill] --KVSender/NIXL--> [D receive] --> D decode --> tool call 출력
     |                                              |
     └ release_kv_cache() → radix LRU 대상          └ 요청 종료 후 KV free
                                                       (decode offload 켜면 host/storage로)

     ↑ tool-call 유휴시간: 클라이언트가 tool 실행 → 결과 반환까지 서버는 이 대화에 대해 idle

Turn N+1:
  [P prefill 다시] ← 누적 대화이력 전체를 재-prefill
                     (P radix에 prefix가 남아있지 않으면 전부 recompute → TTFT 증가)
```

### 1.2 관찰된 문제 (experiments/index.md)

- Agentic multi-turn은 **prefill-heavy**: 매 턴 system prompt + tool 정의 + 누적 대화이력이 context로 전달된다.
- KV cache 점유가 **D node에 집중**된다. 실측(vllm-ppd C=8): D node가 P node 대비 약 7배 높은 KV 점유(D max 17.2% vs P max 3.1%).
- 즉 **P node GPU가 유휴 상태**인데, D node는 요청 종료 시 대화 prefix KV를 버린다. 다음 턴에 P가 이 prefix를 다시 계산한다.

### 1.3 핵심 아이디어

Tool-call 유휴시간 동안, D가 버릴 대화 prefix KV를 **유휴 P node의 GPU radix cache로 옮겨둔다(park)**. 다음 턴 요청이 cache-aware routing으로 같은 P node에 도달하면 **prefix hit**이 발생해 재-prefill을 스킵한다. GPU 용량이 부족하면 P의 CPU DRAM(host pool, Tier3)으로 강등하고, 다음 턴에 load back한다.

### 1.4 왜 로컬 host L2가 아니라 원격 P GPU인가 (NVLink)

Phase 0 실측(→ `idle_kv_parking_phase0.md` §5)에서 **로컬 host DRAM L2**(P가 자기 prefix를 host로 offload)만으로도 압박+유휴 시나리오에서 fetch가 recompute를 이겼다(radix 2.40s → hicache_host 1.45s). 그렇다면 원격 P GPU parking이 굳이 나은 이유는 **전송 경로**에 있다.

| 방식 | park 경로 | 다음 턴 fetch 경로 | host RAM |
|---|---|---|---|
| **host L2 (현재)** | device→host (PCIe ~26 GB/s, 1홉) | host→device (PCIe ~26 GB/s, 1홉) | 소모 |
| **원격 P GPU parking (본 설계)** | D→P GPU (**NVLink ~56 GB/s**, 1홉) | **0 — P GPU radix에 상주해 prefix-hit** | 미소모 |

- **server17 실측 확정**: `nvidia-smi topo -m`에서 P(GPU0)↔D(GPU1) = `NV4`, `nvidia-smi nvlink -s`에서 링크당 14.062 GB/s × 4 = **56.25 GB/s(단방향)**. PCIe Gen4(~26 GB/s)의 **약 2.1배**, 지연도 낮다.
- **KV-transfer 마이크로벤치 실측**(`experiments/benchmark/nvlink_kv_transfer_microbench.py`, P2P=True): NVLink park **52.3–52.8 GB/s**(스펙의 94%) vs host device↔host **26.2–26.4 GB/s**. 512MB(4096토큰) 기준 **NVLink park 10.2ms(fetch 0)** vs **host 왕복 40.8ms(park 20 + fetch 20)** → **~4.0배** 일관. 둘 다 재계산(prefill 수백 ms) 대비 무시 가능하고, park는 유휴시간에 발생해 critical path 밖.
- 더 큰 이점은 **fetch 전송이 0**이라는 것: parked KV가 이미 P GPU에 있으므로 다음 턴 prefill이 그 자리에서 prefix-hit → host L2처럼 PCIe로 되읽을 필요가 없다.
- decode 노드의 **빠듯한 host RAM을 소모하지 않는다**(실험 환경 제약).
- **tier 순서**: P-GPU(NVLink, 빠르고 작음) → host DRAM → disk. parked 엔트리는 P 자체 prefill 압박 시 evict 가능해야 한다.
- ⚠️ **GPU 배치 제약**: NVLink bridge는 server17에서 (GPU0-1), (GPU2-3) 쌍에만 존재. 1P1D(P=0/D=1)는 NVLink ✅. **2P2D로 확장 시 현재 배치(P1=0,P2=1,D1=2,D2=3)는 P→D가 bridge를 가로질러 PCIe가 되므로**, NVLink parking을 쓰려면 배치를 NVLink 쌍 기준으로 재조정해야 한다: `P1=0/D1=1`, `P2=2/D2=3`.

---

## 2. 기존 코드 자산 (재사용 대상)

이 기능은 sglang에 이미 존재하는 hierarchical cache / decode offload 기계를 최대한 재사용한다.

| 구성요소 | 위치 | 역할 |
|---|---|---|
| `DecodeKVCacheOffloadManager` | `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py` | D node가 요청 종료 후 KV를 device→host→storage로 offload (prefix hash keyed). **본 기능의 D-side 패턴 원본.** |
| `DecodeHiCache*Mixin` | `python/sglang/srt/disaggregation/decode_hicache_mixin.py` | 다음 요청 시 host/storage에서 load-back(fetch) |
| `HiCacheController` | `python/sglang/srt/managers/cache_controller.py` | `write`(GPU→host) / `load`(host→GPU) / `prefetch` / `backup`(host→storage) 스레드 |
| `HiRadixCache` + `MHATokenToKVPoolHost` | `python/sglang/srt/mem_cache/hiradix_cache.py`, `memory_pool_host.py` | L1(GPU) ↔ L2(CPU DRAM) ↔ L3(storage) prefix cache |
| `NixlKVManager` / `NixlKVSender` / `NixlKVReceiver` | `python/sglang/srt/disaggregation/nixl/conn.py` | NIXL agent-to-agent 전송(VRAM+DRAM 등록, staging room 지원). **역방향 D→P 채널의 기반.** |
| Decode offload 활성화 플래그 | `python/sglang/srt/server_args.py:863` `--disaggregation-decode-enable-offload-kvcache` | (현재는 `hicache-storage-backend` 필요) |

**결론**
- "Tier3 CPU DRAM fetch" 요구는 기존 decode offload / hicache 경로의 **확장·튜닝**에 가깝다 (Phase 0에서 baseline 측정).
- "유휴 P node로 D→P KV 전송 후 fetch"가 **신규 부분**이다. 현재 KV 전송은 `KVSender`(prefill) → `KVReceiver`(decode) **단방향(P→D)만** 존재한다 (`python/sglang/srt/disaggregation/base/conn.py`). 역방향(D→P)과 "remote P GPU tier" 개념이 없다.

---

## 3. 설계 결정 (확정)

| 항목 | 결정 | 근거 |
|---|---|---|
| **Park 대상 계층** | **Design A: P GPU radix** (초과 시 CPU DRAM 강등) | 다음 턴 prefix hit로 재-prefill 스킵 → TTFT 이득 최대화 |
| **라우팅 affinity** | **기존 router cache-aware routing에 의존** | sgl-router의 prefix cache-aware routing이 같은 P로 보내길 기대. sglang python만 수정, router 무변경. 가장 빠른 실험 착수 |
| **전송 백엔드** | **NIXL** | agent 대칭 구조라 역방향(D→P) 구현이 Mooncake보다 자연스러움 |

---

## 4. 목표 데이터 흐름

```
Turn N 종료 (D idle):
  D: release_kv_cache 직전
     → IdleKVParkManager.push(prefix_hash, kv_indices)
       --NIXL (D→P)-->  P: ParkReceiver.receive()
                            → HiRadixCache.insert (GPU radix, locked)
                            → GPU 용량 초과 시: HiCacheController.write → host pool (Tier3)

Turn N+1:
  router (cache-aware) → 같은 P node
     → prefix match hit (GPU radix에 park된 prefix)
     → prefill 스킵 / 부분 prefill
     → 기존 P→D 경로로 KV 전송 → D decode
```

---

## 5. 컴포넌트 및 수정 지점

### 5.1 신규 모듈: `disaggregation/idle_kv_parking.py`

- `IdleKVParkManager` — `DecodeKVCacheOffloadManager` 패턴 복제.
  - **D-side**: `push(req)` — 종료된 요청의 prefix KV를 prefix-hash로 키잉하여 P로 전송 큐에 넣음. 진행상태 추적(`ongoing_park`), ack 처리.
  - **P-side**: `receive_and_insert()` — 도착한 KV를 `HiRadixCache`에 insert + lock, 용량 초과 시 host pool 강등.
- 유휴 판정 및 용량 상한(park가 활성 prefill을 밀어내지 않도록) 로직 포함.

### 5.2 NIXL 역방향 채널: `disaggregation/nixl/conn.py`

- `ParkSender`(D) / `ParkReceiver`(P) 추가. 기존 `NixlKVManager.send_kvcache` + staging room 인프라 재사용.
- **prefix-hash 기반 park bootstrap room** 신설 (현재는 per-request `bootstrap_room`만 존재). park 세션의 lifecycle/bootstrap 설계 필요.
- 참고: `base/conn.py`, `common/conn.py`의 `BaseKVSender`/`BaseKVReceiver` 계약과 정합.

### 5.3 Scheduler 배선: `managers/scheduler.py` (~L475)

- P/D 양쪽에 `IdleKVParkManager` 인스턴스화 (기존 `decode_offload_manager` 옆).
- ⚠️ **`Scheduler.__init__` 수정 → `large-class-init-style` 스킬을 먼저 읽고 규칙을 따를 것.**

### 5.4 Decode 트리거: `disaggregation/decode.py` (~L1890, 요청 종료 경로)

- KV free 직전에 park 큐로 enqueue (free 대체 또는 병행).
- 유휴 판정은 `scheduler.py:3363` 부근 idle 로직 및 `decode_offload_manager.ongoing_offload` 패턴 참고.

### 5.5 Prefill 수신·삽입: `mem_cache/hiradix_cache.py` / `radix_cache.py`

- P가 받은 KV를 GPU radix tree에 insert + lock, prefix match로 노출.
- 용량 부족 시 `HiCacheController.write`로 host pool(Tier3) 강등, 다음 턴 `load` back.
- park 엔트리의 eviction 우선순위/lock 정책(활성 prefill 보호) 정의.

### 5.6 플래그 / 환경변수: `server_args.py`, `environ.py`

- `--enable-idle-kv-parking`, park 용량 상한/임계값, 전송 stride 등.
- ⚠️ **`SGLANG_*` env var 추가 → `env-var-conventions` 스킬을 먼저 읽고 규칙을 따를 것.**

### 5.7 관측 지표: `disaggregation/kv_events.py` 등

- park hit rate, parked tokens, host 강등량, TTFT 개선.
- BFCL 2P2D 벤치로 검증 (experiments/ 연동).

---

## 6. 단계별 구현 계획

### Phase 0 — 기존 경로 baseline (코드 수정 최소) → 런북: [`idle_kv_parking_phase0.md`](./idle_kv_parking_phase0.md)
- `--disaggregation-decode-enable-offload-kvcache` + hicache로 **Tier3(CPU DRAM/storage) fetch가 어디까지 동작하는지 1P1D 벤치로 측정**.
- 이 결과가 baseline이자 "Tier3" 요구의 기존 커버리지. host-only 모드(storage backend 없이) 허용 완화가 필요한지 판단.
- **조사 완료**: Prefill L2(CPU DRAM) 재사용은 `--enable-hierarchical-cache`만으로 host-only 가능 → 완화 불필요. Decode offload만 storage backend 강제(`server_args.py:4346`). 측정 하니스는 experiments 저장소(`scripts/sglang/run_phase0_baseline.sh` 등)에 구현됨.

### Phase 1 — D→P GPU parking 뼈대 (핵심)
1. `idle_kv_parking.py` 스켈레톤 (`IdleKVParkManager`).
2. NIXL 역방향 채널 (`ParkSender`/`ParkReceiver`, park room bootstrap).
3. Scheduler 배선 (`large-class-init-style` 준수).
4. Decode 트리거 (free 직전 park enqueue).
5. Prefill 수신·insert (`HiRadixCache`).
6. 플래그/env (`env-var-conventions` 준수).
7. **1P1D로 먼저 증명** (affinity 문제 제거).

### Phase 2 — Tier3 강등
- P GPU 용량 부족 시 host pool 강등 + 다음 턴 load back. 기존 hicache 재사용.

### Phase 3 — 관측·검증
- park hit rate / parked tokens / TTFT 메트릭.
- BFCL 2P2D로 TTFT 개선 측정 (cache-aware routing affinity 실증).

---

## 7. 리스크 및 열린 질문

| 리스크 | 설명 | 완화책 |
|---|---|---|
| **Router affinity 미보장** | cache-aware routing이 park 보유 P로 다음 턴을 안 보내면 park 무용 | Phase 1을 **1P1D로 먼저 증명**. 2P2D에서 affinity hit rate 측정 후 필요시 session-sticky 도입 |
| **P GPU eviction 경쟁** | park KV가 P의 활성 prefill을 밀어내면 역효과 | park 엔트리 용량 상한 + lock/우선순위 정책 |
| **NIXL 역방향 lifecycle** | agent 대칭이라 전송은 가능하나 park room bootstrap/정리는 신규 설계 | per-request room 대신 prefix-hash room, TTL/정리 로직 명시 |
| **Prefix 무효화** | 다음 턴이 park된 prefix와 실제로 일치하지 않으면(대화 분기) 낭비 | prefix-hash 기반 매칭으로 자연 무효화, park 히트율 관측 |

---

## 8. 관련 스킬 (수정 전 필독)

- `Scheduler.__init__` 수정 → **`large-class-init-style`**
- `SGLANG_*` env var 추가/변경 → **`env-var-conventions`**

---

## 9. 관련 연구 및 현실성 평가

아이디어의 학술적 위치(OrbitCache 유추, Mooncake/MemServe/AttentionStore/CacheGen 등 관련 연구), fetch-vs-recompute 정량 검증, 현실성/차별화 평가는 별도 문서 참고: [`idle_kv_parking_related_work.md`](./idle_kv_parking_related_work.md)
