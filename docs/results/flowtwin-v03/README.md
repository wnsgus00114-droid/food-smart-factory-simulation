# FlowTwin v0.3 opened-D2 개발 결과

이 디렉터리는 `ml_results/FlowTwin-v03-D2-development-report`에서 생성한 소형 공개 결과표다. 대용량 checkpoint, test prediction, synthetic dataset과 cache는 저장소에 포함하지 않으며 `FLOWTWIN_GUARD.md`의 명령으로 재생성한다.

| 파일 | 내용 |
|---|---|
| [diagnostic_table.csv](diagnostic_table.csv) | Hybrid, FlowTwin-Guard, TCN, DSPR의 4-model 진단 결과 |
| [operational_feasibility.csv](operational_feasibility.csv) | validation-only 알람 탐색과 운전점 통과 여부 |
| [conditional_operational_table.csv](conditional_operational_table.csv) | validation gate를 통과한 3-model의 조건부 test 운영 결과 |

평가 조건은 W96, seed `20260727`, 이미 열린 합성 `D2-ood-dev`를 사용한 post-hoc 개발 실험이다. 따라서 표는 재현 가능한 개발 기록이지 독립 confirmatory 결과가 아니다. TCN은 validation grid 75개에서 운전점을 찾지 못했으므로 operational test 값을 `0`으로 대체하지 않고 제외했다. 전체 해석과 주장 경계는 [FLOWTWIN_GUARD.md](../../../FLOWTWIN_GUARD.md), [ML_EXPERIMENTS.md](../../../ML_EXPERIMENTS.md), [NOVELTY_EVALUATION.md](../../../NOVELTY_EVALUATION.md)를 따른다.

CSV는 로컬 보고서의 수치와 필드 순서를 보존하고 줄바꿈만 Git 저장소용 LF로 정규화했다. 무결성 해시는 [checksums.sha256](checksums.sha256)에 기록한다.
