. .\malariaenv\Scripts\Activate.ps1

python src/pipeline_b_v2/e2e_eval.py `
    --dataset bbbc041 `
    --img-dir data/malaria/images `
    --ann-csv data/processed/test_annotations.csv `
    --checkpoint Phase3-PipelineB/checkpoints/best.pth `
    --stage1-version v2 `
    --out-dir results/v2/e2e_bbbc041_v2
