# ECE598SG final project repo
## 1. Installation
To install necessary packages you need:

```sh
conda install opencv matplotlib numpy pyyaml
```

To test out dataloader using ms COCO, you need:
```sh
mkdir data
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
```

For quick testing, I suggest that you download validation set (since it is small)
and modify `test_dataloader.py` function to use validation set only.

## 2. Testing Dataloader
to test dataloader, please run:

```sh
python test/test_dataloader.py --config-file=configs/debug-dataload.yaml --dataset-type=0
```

## 3. Extending
You can extend files under scripts / utils to add functionalities.


