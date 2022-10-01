function readURL(input) {

    if (input.files && input.files[0]) {

        var reader = new FileReader();
      
        reader.onload = function (e) {
            // document.getElementById("your-image").src = e.target.result;
            $('#your-image')
                .attr('src', e.target.result)
                .width(700)
                .height(500);
            // alert(e.target.result)
            // document.getElementById("your-image").src = e.target.result;

        };

        // $(input).get(0).files[0];
        // localStorage.setItem("img-key", $(input).get(0).files[0])
        // alert(file)
        reader.readAsDataURL(input.files[0]);
        
    }
    }
    