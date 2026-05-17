. .\malariaenv\Scripts\Activate.ps1

python src/pipeline_b_v2/e2e_eval.py `
    --dataset mpidb `
    --img-dir data/MP-IDB/img `
    --ann-csv data/processed/mpidb_annotations.csv `
    --checkpoint Phase3-PipelineB/checkpoints/best.pth `
    --stage1-version v2 `
    --out-dir results/v2/e2e_mpidb_v2
