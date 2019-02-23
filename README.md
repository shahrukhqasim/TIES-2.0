### To visitors
This is not a complete implementation of anything. Its only a development repository which is open source. I'll remove
this note, if at some point, this repository becomes actually usable.
# TIES-2.0
TIES was my undergraduate thesis, Table Information Extraction System. I picked the name from there and made it 2.0
since this is supposed to be much larger than the original version.

## Development note
The project is divided into language parts, `python` and `cpp`, for python and C++ respectively.

The `python` dir is supposed to be the path where a script is to be run, or alternatively, it could be added to the
`$PYTHONPATH` environmental variable. It would contain further directories:
1. `bin` contain the scripts which are to be run from the terminal. Within bin, there would be multiple folders,
short for different classes of executable programs.
    1. iterate
    2. *to be added*
    3. etc
2. Other folders would mostly be python modules, such as trainers, models etc.

Within the context of this repository, `iterate` refers to any of train, test or anything which is done iteratively. You
can say anything that is done iteratively mostly on the GPU. So if there is an `iterator` somewhere, it probably refers
to an entity which handles training, testing etc.

## Requirements
1. Python 3.5+
2. TensorFlow 1.8+
3. Matplotlib
4. Overrides (`pip install overrides`)
5. `configparser`
5. More added as I install them
