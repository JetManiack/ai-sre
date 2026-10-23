FROM golang:1.26-alpine AS build
RUN apk add --no-cache gcc musl-dev
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -trimpath -o /out/aisre ./cmd/aisre

FROM alpine:3.20
RUN apk add --no-cache ca-certificates tzdata
COPY --from=build /out/aisre /usr/local/bin/aisre
ENTRYPOINT ["/usr/local/bin/aisre"]
CMD ["serve", "--config", "/etc/ai-sre/config.yaml"]
