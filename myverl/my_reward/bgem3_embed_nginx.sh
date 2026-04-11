#!/bin/bash

cd `dirname $0`
curr_ip=$(python get_host_ip.py)

c=0
for port in {50000..50030}; do
    if [ `lsof -i :$port | wc -l` -gt 0 ]; then
        echo "端口 $port 被占用"
    else
        echo "端口 $port 可用"
        # CUDA_VISIBLE_DEVICES="$c" nohup python -u bgem3_embed_server.py --port $port 2>&1 | tee -a $CHECKPOINT_SAVE/bgem3_embed_$RANK.log &
        CUDA_VISIBLE_DEVICES="$c" nohup python -u bgem3_embed_server.py --port $port > /dev/null 2>&1 &
        let c++
        echo "${curr_ip}:$port" >> $CHECKPOINT_SAVE/bgem3_embed_$RANK.ip
        if [ $c -eq 8 ]; then
            break
        fi
    fi
done

sleep 80s

if [ $RANK -eq 0 ]; then
    PORT=$1
    NGINX_FILE=$CHECKPOINT_SAVE/nginx_embed.conf
    
    echo "user root;" > $NGINX_FILE
    echo "worker_processes auto;" >> $NGINX_FILE
    echo "error_log /var/log/nginx/error.log;" >> $NGINX_FILE
    echo "pid /var/run/nginx.pid;" >> $NGINX_FILE
    echo "" >> $NGINX_FILE

    echo "events { " >> $NGINX_FILE
    echo "  worker_connections 128;" >> $NGINX_FILE
    echo "  use epoll;" >> $NGINX_FILE
    echo "}" >> $NGINX_FILE
    echo "" >> $NGINX_FILE

    echo "http { " >> $NGINX_FILE
    echo "  upstream bgem3_embed { " >> $NGINX_FILE

    for file in `ls $CHECKPOINT_SAVE/bgem3_embed_*.ip`; do
        echo "file: $file"
        cat $file | while read line; do
            echo "    server $line;" >> $NGINX_FILE
        done
    done

    echo "  }" >> $NGINX_FILE
    echo "  server { " >> $NGINX_FILE
    echo "    listen $PORT;" >> $NGINX_FILE
    echo "    location /embedding { " >> $NGINX_FILE
    echo "      proxy_pass http://bgem3_embed;" >> $NGINX_FILE
    echo "      proxy_connect_timeout 10s;" >> $NGINX_FILE
    echo "      proxy_read_timeout 10s;" >> $NGINX_FILE
    echo "    }" >> $NGINX_FILE
    echo "  }" >> $NGINX_FILE
    echo "}" >> $NGINX_FILE

    echo "nginx.conf file: $NGINX_FILE"
    nginx -c $NGINX_FILE
fi


