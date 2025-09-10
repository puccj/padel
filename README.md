# Padel analyzer

Padel analysis from a pair of vides of the match. Uses YOLO, custom re-identification logic, shots recognition and classification.

## Dependencies

This project is tested and working with python `3.13`. Required packages to make the code flows are:
- `ultralytics`
- `opencv-python`
- `opencv-contrib-python`
- `lapx`

## Getting Started

(Optional but recommended) Create a conda python environment and activate it:
```
conda create -n padel python
conda activate padel
```

Clone the repository:

```
git clone https://github.com/puccj/padel
cd padel
```

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```

## Notes

Folder `src` contains older attemps in C++.