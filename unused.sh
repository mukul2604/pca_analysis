brew services stop mongodb
brew services start mongodb
mongoimport -d lr -c lrecords --type csv --file data/Letter_recognition.csv --headerline
