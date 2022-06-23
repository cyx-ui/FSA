# A Fourier-based Semantic Augmentation for Visible-Thermal Person Re-Identification 
Pytorch Code of FSA method for Cross-Modality Person Re-Identification (Visible Thermal Re-ID) on RegDB dataset and SYSU-MM01 dataset.
*Both of these two datasets may have some fluctuation due to random spliting.

### 1. Prepare the datasets.

- (1) RegDB Dataset : The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

- (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

- A private download link can be requested via sending me an email (mangye16@gmail.com). 

- (2) SYSU-MM01 Dataset : The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

- run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Joint Training.
Train a model by
```bash
python train_ext.py --dataset sysu --lr 0.1 --batch-size 6 --num_pos 4 --fsa_method FSA --lam 0.8 --gpu 0
```

- `--dataset`: which dataset "sysu" or "regdb".

- `--lr`: initial learning rate.

- `--gpu`:  which gpu to run.
  
- `--fsa_method`: which semantic augmentation method to use.

You may need mannully define the data path first.

**Parameters**: More parameters can be found in the script.

**Sampling Strategy**: N (= bacth size) person identities are randomly sampled at each step, then randomly select four visible and four thermal image.

**Training Log**: The training log will be saved in `log/" dataset_name"+ log`. Model will be saved in `save_model/`.

### 3. Testing.

Test a model on SYSU-MM01 or RegDB dataset by using testing augmentation with HorizontalFlip
```bash
python testa.py --mode all --resume 'model_path' --gpu 0 --dataset sysu
```
- `--dataset`: which dataset "sysu" or "regdb".

- `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).

- `--trial`: testing trial (only for RegDB dataset).

- `--resume`: the saved model path.

- `--gpu`:  which gpu to run.

