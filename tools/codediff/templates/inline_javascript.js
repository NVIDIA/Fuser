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
