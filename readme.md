# File Upload
Invoke-RestMethod -Uri "http://127.0.0.1:8000/upload" `
>>   -Method POST `
>>   -ContentType "application/json" `
>>   -Body '{"path": "C:/Users/hp/Downloads/doc.pdf"}'

# One Liner
Invoke-RestMethod -Uri "http://127.0.0.1:8000/upload" -Method POST -ContentType "application/json" -Body '{"path": "C:/Users/hp/Downloads/doc.pdf"}'


# Query
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"query": "Where is Ihifix located?"}'

# One Liner
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method POST -Headers @{ "Content-Type" = "application/json" } -Body '{"query": "Where is Ihifix located?"}'