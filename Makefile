CFG ?= configs/default.yaml
PY  ?= python
RUN = $(PY) scripts/run_pipeline.py --config $(CFG)

.PHONY: data prepare tune train evaluate figures report all smoke api docker test space

data:      ; $(RUN) --stages download
prepare:   ; $(RUN) --stages prepare
tune:      ; $(RUN) --stages tune_yolo tune_vit
train:     ; $(RUN) --stages train_yolo crops train_vit train_rtdetr
evaluate:  ; $(RUN) --stages evaluate benchmark
figures:   ; $(RUN) --stages explain plots
report:    ; $(RUN) --stages report && cd report && pdflatex RoadVision_IEEE.tex && pdflatex RoadVision_IEEE.tex
all:       ; $(RUN) --stages all
benchmark-china: ; $(RUN) --override configs/china_811.yaml --stages prepare train_yolo crops train_vit train_rtdetr evaluate benchmark
smoke:
	$(PY) scripts/make_synthetic_rdd.py --n 60
	$(RUN) --override configs/smoke.yaml --stages prepare train_yolo crops train_vit train_rtdetr evaluate benchmark plots report
api:       ; uvicorn api.main:app --reload --port 8000
ui:        ; mlflow ui --backend-store-uri ./mlruns --port 5000
docker:    ; docker build -t roadvision . && docker run --rm -p 7860:7860 roadvision
test:      ; PYTHONPATH=src:. pytest -q tests
space:     ; bash deploy/push_to_hf_space.sh
