{#-
SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-#}
function toggleDiv (divId) {
const x = document.getElementById(divId)
if (x.style.display === 'none') {
x.style.display = 'block'
} else {
x.style.display = 'none'
}
}
function toggleOldPreamble () {
const old_div = document.getElementById('old_preamble')
const new_div = document.getElementById('new_preamble')
const diff_div = document.getElementById('preamble_diff')
new_div.style.display = 'none'
diff_div.style.display = 'none'
if (old_div.style.display === 'none') {
old_div.style.display = 'block'
} else {
old_div.style.display = 'none'
}
}
function toggleNewPreamble () {
const old_div = document.getElementById('old_preamble')
const new_div = document.getElementById('new_preamble')
const diff_div = document.getElementById('preamble_diff')
old_div.style.display = 'none'
diff_div.style.display = 'none'
if (new_div.style.display === 'none') {
new_div.style.display = 'block'
} else {
new_div.style.display = 'none'
}
}
function togglePreambleDiff () {
const old_div = document.getElementById('old_preamble')
const new_div = document.getElementById('new_preamble')
const diff_div = document.getElementById('preamble_diff')
old_div.style.display = 'none'
new_div.style.display = 'none'
if (diff_div.style.display === 'none') {
diff_div.style.display = 'block'
} else {
diff_div.style.display = 'none'
}
}
function toggleOldCode (testnum, kernelnum) {
const cuda_button = document.getElementById(`cudabutton_${testnum}_${kernelnum}`)
const ptx_button = document.getElementById(`ptxbutton_${testnum}_${kernelnum}`)
const oldcode_button = document.getElementById(`oldcodebutton_${testnum}_${kernelnum}`)
const diff_button = document.getElementById(`diffbutton_${testnum}_${kernelnum}`)
const newcode_button = document.getElementById(`newcodebutton_${testnum}_${kernelnum}`)
const old_cuda_div = document.getElementById(`oldcode_${testnum}_${kernelnum}`)
const new_cuda_div = document.getElementById(`newcode_${testnum}_${kernelnum}`)
const diff_cuda_div = document.getElementById(`diff_${testnum}_${kernelnum}`)
const old_ptx_div = document.getElementById(`oldptx_${testnum}_${kernelnum}`)
const new_ptx_div = document.getElementById(`newptx_${testnum}_${kernelnum}`)
const diff_ptx_div = document.getElementById(`ptxdiff_${testnum}_${kernelnum}`)
const old_div = cuda_button.disabled ? old_cuda_div : old_ptx_div
const diff_div = cuda_button.disabled ? diff_cuda_div : diff_ptx_div
const new_div = cuda_button.disabled ? new_cuda_div : new_ptx_div
new_div.style.display = 'none'
diff_div.style.display = 'none'
diff_button.classList.remove('selected')
newcode_button.classList.remove('selected')
if (old_div.style.display === 'none') {
oldcode_button.classList.add('selected')
old_div.style.display = 'block'
} else {
oldcode_button.classList.remove('selected')
old_div.style.display = 'none'
}
}
function toggleNewCode (testnum, kernelnum) {
const cuda_button = document.getElementById(`cudabutton_${testnum}_${kernelnum}`)
const ptx_button = document.getElementById(`ptxbutton_${testnum}_${kernelnum}`)
const oldcode_button = document.getElementById(`oldcodebutton_${testnum}_${kernelnum}`)
const diff_button = document.getElementById(`diffbutton_${testnum}_${kernelnum}`)
const newcode_button = document.getElementById(`newcodebutton_${testnum}_${kernelnum}`)
const old_cuda_div = document.getElementById(`oldcode_${testnum}_${kernelnum}`)
const new_cuda_div = document.getElementById(`newcode_${testnum}_${kernelnum}`)
const diff_cuda_div = document.getElementById(`diff_${testnum}_${kernelnum}`)
const old_ptx_div = document.getElementById(`oldptx_${testnum}_${kernelnum}`)
const new_ptx_div = document.getElementById(`newptx_${testnum}_${kernelnum}`)
const diff_ptx_div = document.getElementById(`ptxdiff_${testnum}_${kernelnum}`)
const old_div = cuda_button.disabled ? old_cuda_div : old_ptx_div
const diff_div = cuda_button.disabled ? diff_cuda_div : diff_ptx_div
const new_div = cuda_button.disabled ? new_cuda_div : new_ptx_div
old_div.style.display = 'none'
diff_div.style.display = 'none'
oldcode_button.classList.remove('selected')
diff_button.classList.remove('selected')
if (new_div.style.display === 'none') {
newcode_button.classList.add('selected')
new_div.style.display = 'block'
} else {
newcode_button.classList.remove('selected')
new_div.style.display = 'none'
}
}
function toggleDiff (testnum, kernelnum) {
const cuda_button = document.getElementById(`cudabutton_${testnum}_${kernelnum}`)
const ptx_button = document.getElementById(`ptxbutton_${testnum}_${kernelnum}`)
const oldcode_button = document.getElementById(`oldcodebutton_${testnum}_${kernelnum}`)
const diff_button = document.getElementById(`diffbutton_${testnum}_${kernelnum}`)
const newcode_button = document.getElementById(`newcodebutton_${testnum}_${kernelnum}`)
const old_cuda_div = document.getElementById(`oldcode_${testnum}_${kernelnum}`)
const new_cuda_div = document.getElementById(`newcode_${testnum}_${kernelnum}`)
const diff_cuda_div = document.getElementById(`diff_${testnum}_${kernelnum}`)
const old_ptx_div = document.getElementById(`oldptx_${testnum}_${kernelnum}`)
const new_ptx_div = document.getElementById(`newptx_${testnum}_${kernelnum}`)
const diff_ptx_div = document.getElementById(`ptxdiff_${testnum}_${kernelnum}`)
const old_div = cuda_button.disabled ? old_cuda_div : old_ptx_div
const diff_div = cuda_button.disabled ? diff_cuda_div : diff_ptx_div
const new_div = cuda_button.disabled ? new_cuda_div : new_ptx_div
old_div.style.display = 'none'
new_div.style.display = 'none'
oldcode_button.classList.remove('selected')
newcode_button.classList.remove('selected')
if (diff_div.style.display === 'none') {
diff_button.classList.add('selected')
diff_div.style.display = 'block'
} else {
diff_button.classList.remove('selected')
diff_div.style.display = 'none'
}
}
function toggleAllNewTestCode () {
const all_divs = document.querySelectorAll('[id^="newtestcode_"]')
if (all_divs.length == 0) {
return
}
const hidden = all_divs.item(0).style.display === 'none'
all_divs.forEach((div) => {
div.style.display = hidden ? 'block' : 'none'
})
}
function toggleAllRemovedTestCode () {
const all_divs = document.querySelectorAll('[id^="removedtestcode_"]')
if (all_divs.length == 0) {
return
}
const hidden = all_divs.item(0).style.display === 'none'
all_divs.forEach((div) => {
div.style.display = hidden ? 'block' : 'none'
})
}
function toggleAllDiffs () {
document.querySelectorAll('[id^="oldcode_"]').forEach((div) => {
div.style.display = 'none'
})
document.querySelectorAll('[id^="newcode_"]').forEach((div) => {
div.style.display = 'none'
})
all_diff_divs = document.querySelectorAll('[id^="diff_"]')
if (all_diff_divs.length == 0) {
return
}
const hidden = all_diff_divs.item(0).style.display === 'none'
all_diff_divs.forEach((div) => {
div.style.display = hidden ? 'block' : 'none'
})
}
function switchToPTX (testnum, kernelnum) {
const cuda_button = document.getElementById(`cudabutton_${testnum}_${kernelnum}`)
const ptx_button = document.getElementById(`ptxbutton_${testnum}_${kernelnum}`)
const old_cuda_div = document.getElementById(`oldcode_${testnum}_${kernelnum}`)
const new_cuda_div = document.getElementById(`newcode_${testnum}_${kernelnum}`)
const diff_cuda_div = document.getElementById(`diff_${testnum}_${kernelnum}`)
const old_ptx_div = document.getElementById(`oldptx_${testnum}_${kernelnum}`)
const new_ptx_div = document.getElementById(`newptx_${testnum}_${kernelnum}`)
const diff_ptx_div = document.getElementById(`ptxdiff_${testnum}_${kernelnum}`)
cuda_button.disabled = false
ptx_button.disabled = true
if (old_cuda_div.style.display === 'block') {
old_cuda_div.style.display = 'none'
old_ptx_div.style.display = 'block'
} else if (new_cuda_div.style.display === 'block') {
new_cuda_div.style.display = 'none'
new_ptx_div.style.display = 'block'
} else if (diff_cuda_div.style.display === 'block') {
diff_cuda_div.style.display = 'none'
diff_ptx_div.style.display = 'block'
}
}
function switchToCUDA (testnum, kernelnum) {
const cuda_button = document.getElementById(`cudabutton_${testnum}_${kernelnum}`)
const ptx_button = document.getElementById(`ptxbutton_${testnum}_${kernelnum}`)
const old_cuda_div = document.getElementById(`oldcode_${testnum}_${kernelnum}`)
const new_cuda_div = document.getElementById(`newcode_${testnum}_${kernelnum}`)
const diff_cuda_div = document.getElementById(`diff_${testnum}_${kernelnum}`)
const old_ptx_div = document.getElementById(`oldptx_${testnum}_${kernelnum}`)
const new_ptx_div = document.getElementById(`newptx_${testnum}_${kernelnum}`)
const diff_ptx_div = document.getElementById(`ptxdiff_${testnum}_${kernelnum}`)
cuda_button.disabled = true
ptx_button.disabled = false
if (old_ptx_div.style.display === 'block') {
old_ptx_div.style.display = 'none'
old_cuda_div.style.display = 'block'
} else if (new_ptx_div.style.display === 'block') {
new_ptx_div.style.display = 'none'
new_cuda_div.style.display = 'block'
} else if (diff_ptx_div.style.display === 'block') {
diff_ptx_div.style.display = 'none'
diff_cuda_div.style.display = 'block'
}
}

async function explainDifference(testnum, kernelnum) {
    const explainButton = document.getElementById(`explainbutton_${testnum}_${kernelnum}`)
    const explanationDiv = document.getElementById(`explanation_${testnum}_${kernelnum}`)
    const explanationContent = document.getElementById(`explanation_content_${testnum}_${kernelnum}`)

    // Get the code blocks
    const oldCodeDiv = document.getElementById(`oldcode_${testnum}_${kernelnum}`)
    const newCodeDiv = document.getElementById(`newcode_${testnum}_${kernelnum}`)

    if (!oldCodeDiv || !newCodeDiv) {
        alert('Error: Could not find code blocks for this kernel.')
        return
    }

    // Extract the actual code content (skip the <pre><code> wrapper)
    const oldCode = oldCodeDiv.querySelector('code').textContent
    const newCode = newCodeDiv.querySelector('code').textContent

    // Show the explanation div and update button state
    explanationDiv.style.display = 'block'
    explainButton.disabled = true
    explainButton.textContent = 'Loading...'
    explanationContent.textContent = 'Analyzing code differences...'

    try {
        // Call the backend service
        const response = await fetch('{{ explain_api_url | default("/api/explain-diff") }}', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                oldCode: oldCode,
                newCode: newCode,
                testName: document.querySelector(`#test_${testnum} b`).textContent,
                kernelNumber: kernelnum
            })
        })

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
        }

        const result = await response.json()

        if (!result.success) {
            throw new Error(result.error || 'Backend returned error')
        }

        // Handle the structured response
        let explanationText = ''

        if (result.summaries && result.summaries.length > 0) {
            // Display AI summaries - each summary has id and summary fields
            explanationText = result.summaries.map((item, index) => {
                if (item.id) {
                    return `${item.id}\n\n${item.summary}`
                } else {
                    return item.summary
                }
            }).join('\n\n')
        } else if (result.diffs && result.diffs.length > 0) {
            // If we have diffs but no summaries, show basic info
            explanationText = `Found ${result.total_changes} differences, but no AI summary available.`
        } else {
            explanationText = 'No differences detected between the code versions.'
        }

        explanationContent.textContent = explanationText || 'No explanation provided.'

    } catch (error) {
        console.error('Error calling explain service:', error)
        explanationContent.textContent = `Error: Failed to get explanation. ${error.message}`
    } finally {
        // Reset button state
        explainButton.disabled = false
        explainButton.textContent = 'Explain'
    }
}

function hideExplanation(testnum, kernelnum) {
    const explanationDiv = document.getElementById(`explanation_${testnum}_${kernelnum}`)
    explanationDiv.style.display = 'none'
}
