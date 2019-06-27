#!/bin/sh
tmpfile=$(mktemp)
head -n 2 $1 > ${tmpfile}
cat ${tmpfile} > $1
rm -f ${tmpfile}
