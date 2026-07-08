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
| **전송 백엔드** | **CUDA IPC + P2P (직접)** ~~NIXL~~ | server17에서 Mooncake는 "No RDMA→TCP fallback"이라 NVLink를 못 쓴다. NIXL도 GPU transport 보장 안 됨. NVLink 이득을 실제로 얻으려면 D↔P GPU 메모리를 CUDA IPC로 공유해 `cudaMemcpyPeer`(P2P)로 직접 옮겨야 한다. |

> **전송 de-risk 실측 (server17)**: **[2a 완료]** 실제 sglang 프로세스에서도 검증: decode가 KV풀 32k+32v IPC 핸들 publish, prefill이 open+P2P read하여 checksum MATCH, KV P2P read ~51.0 GB/s (`Idle KV parking [prefill]: ... verified -> ready for 2b`). sglang allocator에서 `_share_cuda_` 정상. cross-process CUDA IPC + P2P 마이크로벤치(`experiments/benchmark/nvlink_cross_process_p2p_microbench.py`)에서 별개 프로세스 D(GPU1)→P(GPU0) 전송이 **52.2–52.8 GB/s**(단일 프로세스와 동일, IPC 오버헤드 0), `correct=True`. IPC 핸들 교환은 ~150ms 1회성(연결 셋업 시)이라 파킹마다 드는 비용이 아니다. → 슬라이스 2는 CUDA IPC 채널로 구현.

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

### 5.2 CUDA IPC 역방향 채널 (D→P GPU, NVLink)

- **연결 셋업(1회)**: D가 자신의 `token_to_kv_pool` GPU 버퍼의 CUDA IPC 핸들을 P에 전달(제어 채널: 기존 bootstrap/ZMQ 재사용). P가 `cudaIpcOpenMemHandle`로 D의 KV 풀을 자기 주소공간에 매핑.
- **park(유휴 시)**: P가 consumer로서 D의 해당 prefix 페이지들을 자기 KV 풀 페이지로 `copy_`(P2P/NVLink) → HiRadixCache insert. (마이크로벤치의 consumer-pull 패턴과 동형.)
- 페이지 gather: prefix KV는 paged라 페이지 인덱스 목록으로 gather 복사. 기존 decode offload manager의 device→host 페이지 복사 경로를 mirror하되 target을 host가 아닌 **peer GPU(IPC 매핑)**로.
- prefix-hash로 park 항목 식별. 핸들 lifecycle/정리(연결 종료 시 `cudaIpcCloseMemHandle`) 설계.
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
2. CUDA IPC 역방향 채널 (D KV풀 핸들 공유 → P가 peer-copy). de-risk 완료.
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

---

## 9. Phase 1 결과 — Design A(P GPU radix park)는 1P1D에서 실패 (데이터 확정)

슬라이스 2a~3d로 전송·복사·radix insert 파이프라인을 **정확성까지 완비**했다(2a IPC 52.7 GB/s, 2b gather-copy MATCH, 3c insert 200/200 정확). 그러나 실제 이득 측정에서 **파킹은 reuse/TTFT를 개선하지 못했다.**

### 측정 (radix, pool 40000, C=8, TOOL_DELAY=3, BFCL multi-turn)
| | reuse_ratio | avg TTFT | success |
|---|---|---|---|
| 파킹 OFF (A) | 0.392 | 1.848s | 200/200 |
| 파킹 ON (B) | 0.375 | 1.838s | 200/200 |

→ 차이 없음(노이즈). 계측(DIAG)으로 원인 확정:
```
recv=246 processed=30 | skip=0 copy=30 avg-P-had=0.95 | survival=0%
```

### 근본 원인
1. **alloc 실패 88%**: 압박 상태에서 P KV 풀이 꽉 차 파킹 슬롯 할당 불가 → 216/246 드롭.
2. **survival 0%**: 복사된 소수도 hit 전에 즉시 evict.
3. 무압박 시엔 `avg-P-had≈0.99` → P가 이미 prefix 보유 → 파킹은 생성 토큰(~5%)만 추가 = 무의미.

**결론**: "유휴 P GPU" 전제가 압박과 모순한다. P가 evict할 만큼 압박받는 순간(파킹이 필요한 그때) P GPU엔 여유가 없다. **Design A는 병목과 동일 자원(P GPU)을 노려 1P1D에서 구조적으로 무효.** Phase 0에서 실제로 이긴 건 **host DRAM(별도·대용량 tier)**이었다는 사실과 정합.

### Pivot 방향 (후보)
- **P host DRAM로 강등** (원래 아이디어의 Tier3): P GPU alloc 실패 시 host pool(125GB, GPU 풀 때도 여유)로. Phase 0의 검증된 승자. NVLink는 D→P GPU 전송에만 기여, 이후 P GPU→host.
- **진짜 유휴 3번째 GPU**를 park 풀로 (4×A6000에서 GPU2/3). "유휴 GPU spare" 전제를 실제 여유 자원으로 검증. 단 NVLink 쌍(0-1,2-3) 토폴로지 제약.
- **파킹 엔트리 protect(priority)**: evict 방어. 단 active prefill 자원과 trade-off.

### 슬라이스 4 — 유휴 GPU park 풀 (용량 전제는 검증, 하드웨어 이점은 없음)

Design A 실패 원인(P GPU 병목)을 우회하려 **전용 유휴 GPU(GPU2)에 park 풀**을 두었다.

**4a 실측 (pool 200k=26GB @ GPU2, radix, pool 40000, C=8, TOOL_DELAY=3)**
| | Design A (P GPU) | 4a (GPU2 전용 풀) |
|---|---|---|
| 드롭(alloc-fail) | 88% (recv 246→proc 30) | **0%** (recv 260→proc 260) |
| survival | 0% | **100% (32/32)** |

→ **여유 GPU tier는 두 실패모드(alloc-fail·evict)를 모두 제거한다. 용량 전제 검증 완료.**

**그러나 이 하드웨어에서 reuse 이득(4b: GPU2→GPU0 fetch-on-prefill)은 기존 host-DRAM hicache를 넘지 못한다:**
- `nvidia-smi topo -m`: GPU0(P)↔GPU2(park) = `NODE`(PCIe). NVLink 쌍은 (0-1),(2-3)뿐 → park↔fetch 경로가 **PCIe** = host DRAM fetch와 동일 속도.
- GPU2 풀(26GB) < host RAM(125GB) → 용량 열위.
- 결국 4b는 Phase 0의 `hicache_host`(2.40s→1.45s)를 **재현**할 뿐, NVLink 이점은 P↔D에만 존재.

**Phase 1 종합 결론**: "유휴 자원으로 KV parking" 아이디어는 (1) 전송(NVLink P↔D 52GB/s)·(2) 용량(유휴 GPU 100% survival) 각각은 검증됐으나, **이 2×A6000(쌍별 NVLink) 토폴로지에선 두 이점이 한 경로에서 결합되지 않는다** — NVLink는 P-D에만, 여유 GPU는 PCIe로만 접근. 아이디어가 실익을 내려면 **all-to-all NVLink(NVSwitch/DGX)** 또는 **여유 GPU가 P와 NVLink로 연결된 배치**, 혹은 **진짜 multi-node(aggregate GPU memory ≫ single host)** 가 필요하다. 파이프라인(2a~4a) 코드는 그런 환경에서 재사용 가능한 자산으로 남긴다.

## 10. Head-to-head — park(4a) vs host-DRAM hicache (Phase 1 마무리)

§9의 결론("park 저장 티어는 검증됐으나 이 토폴로지에선 host-DRAM hicache를 넘지 못한다")을 **동일 압박에서 back-to-back 수치**로 못박는다. 3개 arm을 한 자리에서(cross-run drift 제거) 측정:

| arm | 구성 | 역할 |
|---|---|---|
| `radix` | GPU-only prefix cache | park의 현실적 base (fetch 통합 없음) |
| `hicache` | + host-DRAM L2 (통합된 fetch 경로) | 이겨야 할 incumbent |
| `park` | radix + 전용 유휴 GPU2 park 풀(4a) | fetch 미통합 저장 티어 |

### 이미 확보된 증거 (기존 phase0p_p40000_c8_d3 delta)

핵심 메커니즘은 이미 데이터에 있다 — **hicache가 이기는 이유는 대역폭이 아니라 "evict된 prefix를 host DRAM에 담아 다시 fetch"하는 통합 경로**다:

| arm | reuse_ratio | uncached(recompute) tok | TTFT(metric) |
|---|---|---|---|
| radix | **0.257** | 2.85M | 2.09s |
| hicache_host | **0.743** | 0.98M | 1.33s |

radix 대비 hicache는 reuse를 **2.9×**(26%→74%) 끌어올려 재계산 토큰을 1/3로 줄인다 → TTFT 2.40s→1.45s.

### 측정 결과 (확정, 3-arm back-to-back, pool 40000, C=8, delay 3s, 200 items)

| arm | TTFT | throughput | reuse_ratio | cached tok | vs radix TTFT |
|---|---|---|---|---|---|
| radix | 1.767s | 64.2 tok/s | **0.399** | 1.42M | — |
| **hicache** | **1.333s** | 66.7 tok/s | **0.744** | 2.64M | **−24.6%** |
| park (4a) | 1.856s | 62.4 tok/s | **0.390** | 1.38M | **+5.0%** |

**예측 그대로 확인됨:**
- `park.reuse (0.390) ≈ radix.reuse (0.399)` ≠ `hicache (0.744)`. park은 KV를 GPU2에 **저장만** 하고 prefill이 읽는 **fetch-on-hit(4b)이 없어 prefix-hit을 만들지 못한다.** (분석기 진단이 자동으로 이를 표기.)
- `park.TTFT (1.856s)`는 hicache(1.333s) 대비 **+39.2% 느리고**, radix보다도 **오히려 +5.0% 느리다** — D→GPU2 파킹 복사 오버헤드가 순손실이다(읽는 쪽이 없으므로).

**결론**: 병목은 대역폭이 아니라 *fetch 통합*이다. 저장 티어를 추가하는 것만으로는 이득이 0이며, 통합된 fetch 경로를 가진 host-DRAM hicache가 명확히 이긴다(reuse 2.9×, TTFT −24.6%). fetch(4b)를 붙여도 이 토폴로지에선 GPU2→GPU0가 PCIe(§9)라 host-DRAM hicache 동률이 상한. → **park 아이디어는 이 2×A6000 하드웨어에서 실익 없음이 메커니즘(reuse 미개선)까지 규명됨.** Phase 1 종결.

### 실행 (turnkey)

```bash
cd ~/experiments
# 3개 arm 자동 순회: start → /metrics before → BFCL → /metrics after → delta → stop
PREFILL_MAX_TOTAL_TOKENS=40000 CONCURRENCY=8 TOOL_DELAY=3 \
  ./scripts/sglang/run_head_to_head.sh
# 결과 표 + 판정:
#   results/head_to_head/h2h_p40000_c8_d3/head_to_head_summary.json
```

`benchmark/head_to_head_analyze.py`가 각 arm의 벤치 summary(TTFT/TPOT/throughput)와 reuse delta(reuse_ratio/cached/L2_used)를 하나의 표로 병합하고, `park vs hicache` TTFT 격차 + `park.reuse ≈ radix.reuse` 진단을 출력한다.

**이 head-to-head가 확인하면** Phase 1은 "park 아이디어는 이 하드웨어에서 host-DRAM hicache 대비 실익 없음"을 *메커니즘(reuse 미개선)*까지 규명하고 닫힌다. 아이디어 자체의 가치는 §9의 토폴로지 조건(all-to-all NVLink / NVLink-paired spare / multi-node)에서만 실현된다.

## 11. Slice 4b — fetch-on-hit 구현 (GPU2 저장 → 다음 turn P로 fetch)

§10의 park(4a)는 **저장만** 해서 reuse가 radix와 같았다(무이득). "PCIe라도 recompute보다 빠를 것"이라는 가설을 실제로 측정하려면 **fetch 경로**가 필요하다 — 이것을 구현했다(slice 4b).

**동작**: prefill 노드가 새 요청을 큐에 넣기 직전(`scheduler._add_request_to_queue`의 PREFILL 분기), `IdleKVParkManager.maybe_fetch(req)`가:
1. 요청 token_ids의 **가장 긴 parked prefix**를 park 인덱스에서 찾고(`_match_park_prefix`: 저장된 각 길이 L에 대해 `hash(req[:L])` 조회, 내림차순 → 최장 hit; `ent[1]==L`로 해시충돌 방어),
2. P가 이미 가진 부분(`match_prefix`)을 빼고 **부족한 tail만** park 풀(GPU2)에서 로컬 KV 풀(GPU0)로 gather-copy(`_gather_copy_park_to_local`, GPU2→GPU0 = 이 토폴로지선 PCIe),
3. radix tree에 insert → 이후 스케줄러의 `match_prefix`가 **prefix-hit** → prefill이 그만큼 재계산을 건너뜀 → TTFT↓, `cached_tokens`↑.

race 안전: fetch·park 모두 스케줄러 **메인 스레드**에서만 인덱스를 만짐(ZMQ recv 스레드는 큐 적재만). park는 D 시점의 KV를 GPU2로 **복사 스냅샷**하므로 D의 슬롯 재사용과 무관.

**수정 파일**: `disaggregation/idle_kv_parking.py`(`maybe_fetch`, `_match_park_prefix`,
`_gather_copy_park_to_local`, `_park_lens` 유지, fetch DIAG), `managers/scheduler.py`(PREFILL 분기 훅).

**검증**: prefix-match 알고리즘(최장 prefix·충돌 방어·wrap eviction)을 standalone 단위테스트로 통과.
prefill 로그의 `GPU%d-fetch` / DIAG의 `FETCH: hits/tok/avg-ms | miss/already/nospace`로 실동작 관측.

### 측정 결과 (pool 40000 = 강압박, C=8, delay 3s, 200 items)

**초기 버그**: 처음엔 `origin_input_ids + output_ids`를 park key로 저장 → tool-call turn에서
클라이언트가 assistant 메시지를 template로 재렌더링하면 raw 생성 토큰과 어긋나 prefix-match
거의 실패(reuse 0.39→0.41, 무개선). **fix: 프롬프트(`origin_input_ids`)만 park** — 이건 다음
turn의 token-exact prefix라 radix/hicache가 매칭하는 단위와 동일.

fix 후:

| arm | reuse_ratio | TTFT | vs radix |
|---|---|---|---|
| radix (GPU prefix cache) | 0.389 | 1.832s | — |
| **park (fetch)** | **0.450** | 1.829s | **−0.2%** (무승부) |
| hicache | 0.744 | 1.381s | −24.6% |

> **radix는 "재계산 baseline"이 아니다** — RadixAttention prefix cache라 GPU 잔존 prefix는 hit한다
> (reuse 0.39). radix가 recompute하는 건 hit 못한 토큰뿐: ①매 turn 새 토큰(~26% floor, 어떤 캐시도
> 불가피) + ②축출된 prefix. parking이 겨냥하는 건 ②뿐이고, ②는 강압박(pool 40k)에서만 존재한다.

→ **fetch-on-hit은 실제로 동작**(reuse 0.39→0.45, +217k 토큰이 재계산 대신 fetch됨). **그러나 순
TTFT 이득은 ~0.** DIAG가 원인을 특정:
```
FETCH: hits=26 tok=95009 avg=235.8ms | miss=206 already=278 nospace=237 (of 747)
survival=100% (32/32)  avg-P-had=0.00
```
- **nospace 32%**: fetch한 KV는 attention이 읽으려면 **압박받는 P GPU 풀에 다시 넣어야** 하는데,
  pool 40000이 꽉 차 `alloc(n)` 실패 → 3분의 1이 stage 불가. **저장은 유휴 GPU로 offload해도
  restore는 병목(P GPU)을 점유해야 한다** — Design A의 병목이 restore 쪽에서 재발.
- **already 37%**: P가 아직 prefix 보유(evict 안 됨) → fetch 불필요(정상).
- **hits 3.5%(26)**: "P가 evict했고 && parked됐고 && 자리 있음" 창이 매우 좁음.
- **avg 235ms/fetch**: 32-layer 동기 gather-copy(GPU2→GPU0). 현 hit 수(26)에선 총 6s(전체의 1.3%)라
  묻히지만, hit이 늘면 병목이 된다.

**hicache가 이기는 이유(0.74 vs 0.45)**: 같은 "P GPU로 restore" 제약을 받지만 (a) 풀 차면 LRU
evict로 자리 확보, (b) **async** prefetch. 본 구현은 (a) alloc 실패시 포기, (b) 동기 복사.

**해석**: "PCIe fetch > recompute"는 **reuse 레벨에선 참**이나, 이 강압박 워크로드에선 회수량이
작고(nospace가 막음) 동기 복사가 있어 **순 TTFT는 무승부**. 근본 한계는 **restore가 병목 P GPU를
점유**해야 한다는 것.

### 압박 완화 스윕 결과 — catch-22 실증 (결정적)

P 풀을 키워 restore 자리를 주고 재측정(`run_head_to_head_pool_sweep.sh`):

| pool | radix TTFT / reuse | hicache TTFT / reuse | park TTFT / reuse | park vs radix |
|---|---|---|---|---|
| 40000 (강압박) | 1.83 / **0.39** | 1.38 / **0.74** | 1.83 / 0.45 | +0.2% |
| 60000 | 1.27 / **0.74** | 1.38 / 0.74 | 1.19 / 0.74 | −6.3%* |
| 80000 | 1.17 / **0.74** | 1.38 / 0.75 | 1.19 / 0.74 | +1.0% |
| 120000 | 1.15 / **0.74** | 1.27 / 0.75 | 1.15 / 0.74 | +0.4% |

\* pool-60000의 −6.3%는 park이 빨라서가 아니라 **radix-60000 TTFT가 outlier(1.27, 다른 pool보다 높음)**
라 생긴 착시다. reuse가 radix(0.736)≈park(0.741)로 사실상 동일 → fetch가 유의미하게 안 걸림. 노이즈.

**핵심 관찰 — 두 조건이 상호배타적(catch-22):**
1. **회수할 가치가 있는 구간 = pool 40000(강압박)뿐.** 여기서만 radix가 evict해 reuse가
   0.39로 떨어지고 hicache(0.74)와 격차가 벌어진다. 그런데 이 구간은 **nospace 32%** — restore가
   P GPU에 자리를 못 잡는 바로 그 구간.
2. **restore 자리가 있는 구간 = pool ≥60000.** 그런데 여기선 radix가 evict를 안 해 **reuse가
   이미 0.74** = hicache와 동일 → **되찾을 게 없다.** 세 arm 모두 reuse 0.74로 수렴.
3. → **"restore가 가치 있다"(evict 발생)와 "restore가 자리 있다"(P 여유)가 결코 공존하지 않는다.**
   park reuse는 모든 pool에서 radix와 동일 → **어떤 operating point에서도 park이 radix를 유의미하게
   이기지 못한다.**

**부가 관찰**: pool ≥60000에서 **hicache가 radix보다 느리다**(TTFT +8~10%). evict가 없으니
host-offload 계층은 순수 오버헤드. hicache의 우위는 오직 강압박(pool 40000)에서만 성립.

### Phase 1 최종 결론 (idle KV parking, 2×A6000 단일 노드)

fetch-on-hit(4b)까지 완비해 end-to-end로 검증한 결과, **park+fetch는 이 하드웨어의 어느
operating point에서도 radix(GPU prefix cache) 대비 순 이득이 없다.** 근본 원인은 구현 디테일이 아니라
구조적 catch-22다:
- 파킹의 가치 = P가 prefix를 evict할 때(압박) 재계산을 fetch로 대체하는 것.
- 그러나 **fetch한 KV는 attention이 읽으려면 병목인 P GPU 풀에 다시 들어가야 한다**(nospace) —
  transfer를 아무리 빨리(NVLink) 해도 이 제약은 **토폴로지 독립적**. 저장은 유휴 자원으로 offload
  되지만 **restore는 병목을 점유**한다.
- 압박을 풀어 자리를 주면 evict 자체가 사라져 회수 대상이 소멸.

hicache가 강압박에서 이기는 이유는 같은 "P GPU로 restore" 제약을 **evict-to-room + async
prefetch**로 관리하기 때문. park을 그 수준으로 엔지니어링하면(evict-to-fetch + async 복사)
hicache를 **재현**할 수 있으나, 이 토폴로지선 GPU2→GPU0가 PCIe(=host DRAM와 동속)이고 용량도
26GB<125GB라 **넘어설 수는 없다.**

**아이디어가 실익을 내려면**: (1) attention이 remote KV를 직접 읽는 **disaggregated/remote
attention**(restore가 P GPU를 점유하지 않아도 됨), 또는 (2) host DRAM이 유일 로컬 tier이고 원격
GPU 합산 용량이 host를 압도하는 **진짜 multi-node**. 단일 노드 단일 GPU-tier로는 hicache가 이미
상한. 파이프라인 코드(2a~4b: IPC/P2P/park/fetch)는 그런 환경의 재사용 자산으로 남긴다.
