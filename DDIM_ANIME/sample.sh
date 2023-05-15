# using DDIM, steps are 10, 100, 1000 respectively
python run.py --mode test --steps 10 --eta 0.0
python run.py --mode test --steps 100 --eta 0.0
python run.py --mode test --steps 1000 --eta 0.0

# using DDPM, steps are 10, 100, 1000 respectively
python run.py --mode test --steps 10 --eta 1.0
python run.py --mode test --steps 100 --eta 1.0
python run.py --mode test --steps 1000 --eta 1.0

# try some other eta
python run.py --mode test --steps 100 --eta 0.2
python run.py --mode test --steps 100 --eta 0.5
