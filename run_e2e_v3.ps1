. .\malariaenv\Scripts\Activate.ps1

python src/pipeline_b_v2/e2e_eval.py `
    --dataset bbbc041 `
    --img-dir data/malaria/images `
    --ann-csv data/processed/test_annotations.csv `
    --stage1-only `
    --stage1-version v3 `
    --out-dir results/v3/e2e_bbbc041_v3
