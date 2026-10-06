FROM node:22-alpine AS builder
WORKDIR /app
COPY package*.json ./
# skip the "prepare" script (playwright browser install); svelte-kit sync runs during build
RUN npm ci --ignore-scripts
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
EXPOSE 80
