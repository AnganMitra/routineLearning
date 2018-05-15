#!/bin/bash

rm -r -f ../testData/*
rm -r -f ../trainData/*
rm -r -f ../comparisons/csv/*
rm -r -f ../comparisons/plot/*
rm -r -f ../generatedDataVisualisation/*
rm -r -f ../modelParameters/*
rm -r -f ../patternLearning/*
rm -r -f ../modelOutput/*
mkdir ../trainData/generated
mkdir ../testData/generated
python3 runnable.py
python3 detect_pattern.py
