document.addEventListener('DOMContentLoaded', function () {
    const actionSelect = document.querySelector('select[name="action"]');
    const modelSelect = document.querySelector('select[name="model"]');
    const warningMessage = document.getElementById('warning-message');

    function checkCombination() {
        const action = actionSelect.value;
        const model = modelSelect.value;
        if (action !== '---' && model !== '---') {
            warningMessage.style.display = 'block';
        } else {
            warningMessage.style.display = 'none';
        }
    }

    actionSelect.addEventListener('change', checkCombination);
    modelSelect.addEventListener('change', checkCombination);
});