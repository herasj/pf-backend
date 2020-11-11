FROM node:12

WORKDIR /app

COPY *.json ./

COPY .env ./

COPY src src

RUN ["npm","i"]

RUN ["npm", "run","build"]

EXPOSE 3000

CMD ["npm","run","start:prod"]
