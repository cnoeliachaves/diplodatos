#!/bin/sh

SPARK_HOME=$HOME/spark/spark-2.1.1-bin-hadoop2.7

FIN="${1:-"$HOME/spark/zeppelin-0.7.3-bin-all/licenses/"}"
FOUT="${2:-"out_csv_dir"}"

set -x

$SPARK_HOME/bin/spark-submit \
    target/scala-2.11/wordcount_2.11-1.0.jar \
    "$FIN" "$FOUT"

set +x

echo

[ $? -eq 0 ] && cat "$FOUT/part"* | head
