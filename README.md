
# TIES-2.0
TIES was my undergraduate thesis, Table Information Extraction System. I picked the name from there and made it 2.0
 from there.
 
 This is a repository containing source code for the arxiv paper xxxx.xxxxx ([link](https://www.google.com)). This paper has been accepted into 
ICDAR 2019. To cite the paper, use:

```
@article{rethinkingGraphs,
  author    = {Qasim, Shah Rukh and Mahmood, Hassan and Shafait, Faisal},
  title     = {Rethinking Table Parsing using Graph Neural Networks},
  journal   = {Accepted into ICDAR 2019},
  volume    = {abs/xxxx.xxxxx},
  year      = {2019},
  url       = {https://arxiv.org/abs/xxxx.xxxxx},
  archivePrefix = {arXiv},
  eprint    = {xxxx.xxxxx},
}
```

## Note to the visitors
We are still working to improve a few technical details for your convenience. We'll remove this note once we are done. 
Expect them to be done by June 15, 2019. We are also working to improve dataset format for easier understanding.

## Dataset
Partial test dataset can be found [here](https://drive.google.com/drive/folders/18QyBB1pavj_xCsTyCR6XC_AA525nZaVZ?usp=sharing
). We are uploading rest of the dataset. The current format of the dataset is `tfrecords`. This dataset is compatible with
this repository. However, we are working on improving the format; we'll publish once it is done.

## Development note
The project is divided into language parts, `python` and `cpp`, for python and C++ respectively.

The `python` dir is supposed to be the path where a script is to be run, or alternatively, it could be added to the
`$PYTHONPATH` environmental variable. It would contain further directories:
1. `bin` contain the scripts which are to be run from the terminal. Within bin, there would be multiple folders,
short for different classes of executable programs.
    1. iterate
    2. *to be added*
    3. etc
2. `iterators` provides functionality to iterate through the datasets while you are training or testing.
3. `layers` contains basic layers for graph networks
3. More documentation coming

Within the context of this repository, `iterate` refers to any of train, test or anything which is done iteratively. You
can say anything that is done iteratively mostly on the GPU. So if there is an `iterator` somewhere, it probably refers
to an entity which handles training, testing etc.

## Training
1. Make a config file according to the format given in `configs/config.ini.example`.
2. Run training using `python bin/iterate/table_adjacency_parsing path/to/the/config/file config`

You can set the paths to test, train, validation files in the config file.

## Inference
1. `python bin/iterate/table_adjacency_parsing path/to/the/config/file config --test True`
2. The output will be pickle format of python.
 
## Requirements
1. Python 3.5+
2. TensorFlow 1.12
3. Matplotlib
4. Overrides (`pip install overrides`)
5. `configparser`
6. `opencv-python`
7. More added as I install them
