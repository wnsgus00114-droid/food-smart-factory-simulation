# D3-RUL 장기 오염·건강도·잔여수명 파이프라인

`D3-RUL`은 기존 D1 고장분류 데이터와 계약·스크립트·산출물 경로를 공유하지 않는 독립 파이프라인이다. 현재 계약 버전은 `1.1.0`이다. 현재 범위는 **중단 없는 단일 생산 run의 scalar `fouling_index` 열화**다. 생산→CIP→재기동 campaign의 soil·chemical carryover를 이 데이터의 RUL 근거로 사용하지 않는다.

상태: generator/split/auditor/train-only baseline과 축소 smoke가 구현됐다. 이는 미검증 engineering research surrogate의 소프트웨어·데이터 계약이 재현된다는 뜻이며, 공장 설비수명이나 정비주기를 검증했다는 뜻이 아니다. 실제 공정·설비, 미생물학적 안전과 법규 적합성 검증은 모두 미수행이다.

> v3 구분: `lifecycle.py`와 `generate_multicycle_rul_dataset.py`에는 반복 생산/CIP, complete/incomplete cleaning, 교정·부분정비·overhaul·replacement, virtual age와 다중 service interval을 별도 구현했다. 이 구현은 D3 계약을 소급 변경하지 않는다. D3 데이터·split·baseline은 계속 단일 생산 scalar-fouling 전용이고, v3 lifecycle은 아직 ML 성능 benchmark가 아닌 합성 정책 시뮬레이션이다. 바이오 모델은 두 경로 모두 범위에서 제외한다.

## 의미와 단위

- `health_index = 1 - fouling_index`
- `degradation_acceleration_factor`는 fouling 성장률에만 적용한다.
- `equivalent operating time = simulation time × acceleration factor`
- `signals.csv`의 `time_sim_s`와 `operating_time_meter_h = time_sim_s / 3600`은 **가속되지 않은 simulator clock**이다. acceleration은 trajectory/oracle metadata에만 있고 모델 feature에는 없으므로 두 관측 clock의 비율로 acceleration을 복원할 수 없다.
- EOL persistence는 controller와 유체 지연의 실제 수치적 시간축인 simulator seconds로 평가한다.
- 역변환된 equivalent RUL은 scalar fouling surrogate의 연구 라벨이며 실제 공장 수명 보증값이 아니다.

EOL은 기동 오경보를 막기 위해 최소 monitor 시간과 정상 forward persistence를 거친 뒤 다음 조건 중 가장 먼저 완료된 시각이다.

1. `fouling_index >= 0.80`
2. steam valve 90% 이상이 60 simulator seconds 지속
3. matched clean-reference 대비 300초 rolling energy/L이 1.25 이상이고 300초 지속
4. 한 번 정상 forward로 arm된 뒤 forward 상실이 5초 지속

각 완료 조건의 시작·완료시각과 값을 `eol_evidence.csv`에 남긴다.

EOL state machine의 평가 격자는 `signals.csv`/`oracle_labels.csv`에 실제로 내보내는 sample interval이다. 각 출력 행의 `[time_start_sim_s, time_sim_s]`는 0부터 종료시각까지 빈틈없이 이어지며, arm·threshold·persistence·동시 원인 우선순위는 이 행들만으로 재현된다. 따라서 persistence 완료시각의 해상도는 `sample_interval_s`이고 manifest의 `eol_evaluation_resolution_sim_s`에 동일한 값을 기록한다.

## Right censoring

Administrative censor horizon은 물리·noise seed와 다른 `censor_seed`로 시뮬레이션 전에 정한다. EOL이 먼저 오면 `event_observed=1`, censor가 먼저 오면 `event_observed=0`이다.

검열 trajectory에서는 `rul_sim_s`와 `rul_equivalent_s`를 반드시 빈 값으로 둔다. 종료시각을 고장시각으로 간주하거나 RUL 0으로 채우지 않는다. 대신 각 landmark에 다음을 기록한다.

- `time_to_event_or_censor_*`
- `rul_lower_bound_*`

관측 EOL trajectory만 exact RUL을 가진다.

## 산출물

- `profiles.csv`: profile domain과 물리 파라미터
- `trajectories.csv`: seed, acceleration, administrative horizon, event/censor outcome
- `signals.csv`: 관측 가능한 공정·압력·utility 계측 proxy
- `oracle_labels.csv`: fouling/health, EOL, survival duration, exact RUL 또는 censor lower bound
- `eol_evidence.csv`: EOL persistence 근거
- `dataset_schema.json`, `dataset_manifest.json`, `checksums.sha256`

Split은 `trajectory_splits.csv`, `split_manifest.json`과 checksum을 만들고, 감사기는 detached checksum이 붙은 `audit_report.json`을 만든다. 기준선은 `health_predictions.csv`, `rul_predictions.csv`, 결과·manifest와 checksum을 별도 디렉터리에 저장한다.

`heater_power_sensor_kw`와 `pump_power_sensor_kw`는 simulator truth에 독립적인 결정적 계측 noise를 더한 관측 proxy다. 실제 `heat_kw`, `pump_power_kw`, fouling rate, acceleration, EOL/censor 결과는 signal 입력에 넣지 않는다.

## 실행

저장 smoke와 같은 생성 파라미터를 쓰는 재현 명령이다. 네 출력 경로는 실행 전에 존재하지 않거나 비어 있어야 한다.

```bash
python3 generate_d3_rul_dataset.py \
  --output /tmp/d3-rul-v2.2-repro/dataset \
  --dataset-version D3-RUL-v2.2-smoke \
  --profiles 6 --trajectories-per-profile 4 --ood-profiles 1 \
  --dt-s 1 --sample-interval-s 10 \
  --maximum-sim-duration-s 1200 \
  --seed 123 \
  --accelerations 100 \
  --censor-horizons-equivalent-h 8 28

python3 split_d3_rul_dataset.py \
  --dataset /tmp/d3-rul-v2.2-repro/dataset \
  --output /tmp/d3-rul-v2.2-repro/splits \
  --seed 456

python3 audit_d3_rul_dataset.py \
  --dataset /tmp/d3-rul-v2.2-repro/dataset \
  --splits /tmp/d3-rul-v2.2-repro/splits \
  --report /tmp/d3-rul-v2.2-repro/audit_report.json

python3 run_d3_rul_baselines.py \
  --dataset /tmp/d3-rul-v2.2-repro/dataset \
  --splits /tmp/d3-rul-v2.2-repro/splits \
  --output /tmp/d3-rul-v2.2-repro/baselines
```

정규 기본값은 80 profiles × profile당 10 trajectories다. Split은 outcome을 보지 않고 `plant_profile_id` hash로 60/20/20을 정한다. 같은 profile, life family, config와 모든 seed는 한 split에만 존재한다. OOD profile은 ID profile parameter range와 겹치지 않는 명시적 override로 생성하고 `test_ood_profile`로 격리한다.

## 저장 smoke 결과

`d3_results/D3-RUL-v2.2-smoke/`는 위 명령의 고정 산출물이다.

| 항목 | 값 |
|---|---:|
| profiles / trajectories | 6 / 24 (`6 × 4`) |
| 별도 OOD profile | 1 |
| signal / label rows | 990 / 990 |
| 관측 EOL / right-censored | 13 / 11 |
| split trajectories | train-ID 12 / validation-ID 4 / test-ID 4 / test-OOD-profile 4 |
| 감사 | 26/26 PASS |
| RUL baseline landmarks | 175 |

Generator/auditor 버전은 각각 `1.2.0`이다. 감사는 checksum, feature/oracle 분리, ID와 겹치지 않는 OOD parameter override, profile·life-family·seed 격리, 건강도 단조성, observable clock, trace에서 재구성한 EOL arming/threshold/persistence/first cause와 exact/censored RUL 의미를 검사했다. Validation split은 이 작은 smoke에서 event-only라 warning을 보존한다. OOD profile은 구조적으로 격리했지만 이 축소 표본의 OOD 점수를 현장 일반화 성능으로 주장하지 않는다. 건강도와 RUL 기준선의 모든 fit 통계는 train-ID만 사용한다.

## 감사와 기준선

감사기는 다음을 실패 조건으로 다룬다.

- checksum·schema·runtime model 호환성 불일치
- oracle 또는 EOL/censor 정보의 `signals.csv` 유입
- profile/life family/seed/config의 split 중복
- `health != 1 - fouling` 또는 생산 fouling 감소
- event/censor metadata와 terminal row 불일치
- censored RUL의 0 채움 또는 lower bound 누락
- 출력 trace를 독립 재생했을 때 arm·threshold·persistence·최초 원인이 재현되지 않는 EOL evidence

기준선은 train-ID split에서만 다음을 적합한다.

- 상수 및 telemetry ridge 건강도
- train outcome Kaplan–Meier conditional RUL
- train의 **관측 EOL만** 사용한 median EOL-age naive RUL
- 과거 ridge-health slope 외삽 RUL, slope가 유효하지 않으면 Kaplan–Meier fallback

기준선 입력과 예측 CSV의 RUL 시간축은 관측 가능한 simulator seconds다. acceleration을 요구하는 equivalent RUL은 oracle 평가 라벨일 뿐 모델 입력이나 기준선 예측 단위로 사용하지 않는다. 조건부 생존확률이 절반 이하로 내려가지 않아 Kaplan–Meier median이 식별되지 않으면 해당 예측은 빈 값으로 남기며 censor 종료시각으로 채우지 않는다. train split에 관측 EOL이 한 건도 없으면 KM/naive RUL 자체가 식별되지 않은 것으로 보고 실행을 중단한다.

Exact RUL MAE는 관측 EOL에만 계산한다. 검열 trajectory는 exact RUL로 대체하지 않고 trajectory-level concordance 비교에 포함한다. 이 기준선은 연구 재현성 확인용이며 정비 또는 식품안전 interlock이 아니다.
