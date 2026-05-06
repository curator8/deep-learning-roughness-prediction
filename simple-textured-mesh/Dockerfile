FROM node:22-slim

WORKDIR /app

COPY simple-textured-mesh/package*.json ./
RUN npm install

COPY simple-textured-mesh ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--no-open"]
