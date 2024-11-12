document.addEventListener("DOMContentLoaded", function() {
    document.querySelector("form").addEventListener("submit", function(event) {
        var actionSelect = document.getElementById('action-select');
        var modelSelect = document.getElementById('model-select');
        if (actionSelect.value === '---' || modelSelect.value === '---') {
            event.preventDefault();
        }
    });
});
