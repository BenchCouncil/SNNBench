set -x
docker build -t bindsnet:0.1.7 -f Dockerfile.cuda92.pytorch041 $(pwd)
docker build -t bindsnet:0.2.4 -f Dockerfile.cuda92.pytorch10 $(pwd)
docker build -t bindsnet:0.2.5 -f Dockerfile.cuda92.pytorch12 $(pwd)
docker build -t bindsnet:0.3.1 -f Dockerfile.cuda102.pytorch110 $(pwd)