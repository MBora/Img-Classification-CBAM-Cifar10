# IMAGE CLASSIFICATION USING CBAM

## Running the program

On two separate terminals, run the following commands
```ps
cd frontend
docker-compose up -d
```
```ps
cd backend
docker build -t myimage .
docker run -p 8000:8000 myimage
```
This starts the frontend server on [http://localhost:3000](http://localhost:3000) and the backend server on [http://localhost:8000](http://localhost:8000).

## Notes

+ PNG images aren't supported because they have 4 channels (RGB and a transparency channel) whereas the model is trained to support only 3 channels.