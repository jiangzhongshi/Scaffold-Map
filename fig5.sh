echo "Downloading Files"
wget https://cs.nyu.edu/~zhongshi/files/filigree100k_miq.obj
wget https://cs.nyu.edu/~zhongshi/files/thai_statue_miq_b.obj

echo "Building"
mkdir build && pushd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j 4

echo "Running Two Examples in Figure 5"
./scaf_param_bin -m ../filigree100k_miq.obj -g 0 -i 100 
./scaf_param_bin -m ../thai_statue_miq_b.obj -g 0 -i 50

echo "Check _uv.obj files in the root folder, thanks!"

