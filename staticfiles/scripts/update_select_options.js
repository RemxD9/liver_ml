function updateSelectOptions() {
    var actionSelect = document.getElementById('action-select');
    var modelSelect = document.getElementById('model-select');
    var selectedAction = actionSelect.value;

    modelSelect.innerHTML = '';

    var defaultOption = new Option('---', '---');
    modelSelect.add(defaultOption);

    if (selectedAction === 'Поиск болезней') {
        var option1 = new Option('UNET', 'UNET');
        var option2 = new Option('DeepLab', 'DeepLab');
        modelSelect.add(option1);
        modelSelect.add(option2);
    } else if (selectedAction === 'Сегментация') {
        var option1 = new Option('UNET', 'UNET');
        var option2 = new Option('UNET++', 'UNET++');
        var option3 = new Option('DeepLab', 'DeepLab');
        modelSelect.add(option1);
        modelSelect.add(option2);
        modelSelect.add(option3);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    updateSelectOptions();
});