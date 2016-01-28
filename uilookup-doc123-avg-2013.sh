java -Xmx8g -cp ../bin model.UILookupDocAvg123Main -embeddingLength 200 -embeddingFile ../../sa-embedding/yelp-sa-embedding-run/window-prediction-2013-200dms-4  -year 2013 -windowSizeWordLookup1 1 -windowSizeWordLookup2 2 -windowSizeWordLookup3 3 -outputLengthWordLookup 50 -embeddingLengthUserLookup 75 -embeddingLengthItemLookup 75 -classNum 5 -inputDir ../../../yelp-13-14 -roundNum 100 -probThreshold 0.001 -learningRate 0.03 -randomizeBase 0.01 -dumpUserEmbeddingFile user-embedding/2013-doc123-rerun-75dms- -dumpItemEmbeddingFile item-embedding/2013-doc123-rerun-75dms 