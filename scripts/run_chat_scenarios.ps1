param(
    [string]$BaseUrl = "http://127.0.0.1:5000",
    [string]$SessionPrefix = "phase63"
)

# Every scenario gets a distinct conversation thread. Turns within a scenario
# share the same session ID so pending workflows can resume naturally.
$scenarios = @(
    @{ Name = "order-follow-up"; Messages = @("Where is my order?", "ORD-1001") },
    @{ Name = "damaged-order"; Messages = @("My order arrived damaged", "ORD-1001", "The box was crushed and the screen is cracked.") },
    @{ Name = "refund-not-initiated"; Messages = @("Where is my refund?", "ORD-1001") },
    @{ Name = "refund-processing"; Messages = @("What is the refund status?", "ORD-1002") },
    @{ Name = "refund-completed"; Messages = @("Has my refund been processed?", "ORD-1003") },
    @{ Name = "refund-no-record"; Messages = @("Where is my refund?", "ORD-1004") },
    @{ Name = "product-availability"; Messages = @("Check product availability", "Laptop Pro") },
    @{ Name = "product-search"; Messages = @("Search for Laptop Pro") },
    @{ Name = "return-policy"; Messages = @("What is your return policy?") },
    @{ Name = "hindi-order"; Messages = @(( -join @([char]0x092E,[char]0x0947,[char]0x0930,[char]0x093E,[char]0x0020,[char]0x0911,[char]0x0930,[char]0x094D,[char]0x0921,[char]0x0930,[char]0x0020,[char]0x0915,[char]0x0939,[char]0x093E,[char]0x0901,[char]0x0020,[char]0x0939,[char]0x0948,[char]0x003F)), "ORD-1001") },
    @{ Name = "interruption-resumption"; Messages = @("Where is my order?", "Where is my refund?", "ORD-1004", "ORD-1001") },
    @{ Name = "unrelated-question"; Messages = @("Can you explain black holes?") }
)

foreach ($scenario in $scenarios) {
    $sessionId = "$SessionPrefix-$($scenario.Name)"
    Write-Host "`n--- $($scenario.Name) / session $sessionId ---"
    foreach ($message in $scenario.Messages) {
        $body = @{ session_id = $sessionId; message = $message } | ConvertTo-Json -Depth 4 -Compress
        try {
            $response = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/chat" -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
            Write-Host "Customer: $message"
            Write-Host "Assistant: $($response.answer)"
        }
        catch {
            Write-Warning "Request failed for '$message': $($_.Exception.Message)"
            break
        }
    }
}
