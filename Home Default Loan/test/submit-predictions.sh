#!/bin/bash

GRADER_API_KEY=`cat GRADER_API_KEY`
API_HEADER="x-api-key: $GRADER_API_KEY"
PREDICTIONS=$1
PART=$2

curl -X POST -H "$API_HEADER" -H "Content-Type: application/json" \
	-d @$PREDICTIONS https://grader.learnml.cool/r24/final-projects/$PART