# To activate correct conda environment. TODO: Refactor
eval "$(conda shell.bash hook)"
conda activate petab_sciml
export PYTHONPATH="$pwd:$PYTHONPATH"

# Run all PyTorch scripts
for ((i = 1 ; i < 52 ; i++)); do
    echo "Test case $i"
    if [ $i -lt 10 ]; then
        path="./test_cases/net_import/00$i/create_testdata/net.py"
    else
        path="./test_cases/net_import/0$i/create_testdata/net.py"
    fi
    python $path
done

exit 0
