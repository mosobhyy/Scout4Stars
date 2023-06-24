function enableFileInput(checkbox, fileInputId) {
    var fileInput = document.getElementById(fileInputId);
  
    if (checkbox.checked) {
        fileInput.disabled = false;
      } 
    else {
        fileInput.disabled = true;

      }
    }
  