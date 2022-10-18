/** 
 * This file is used to store JavaScript functions needed for the web application functionality
 **/

/**
 * Reading the file image input & 
 * changing the 'src' element of the img element (id "your-image") to the source file path of the 
 * image input
 * @param  {File}   input    File input uploaded by the user
 */
function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
      
        reader.onload = function (e) {
            $('#your-image')
                .attr('src', e.target.result)
                .width(700)
                .height(500);
        };
        reader.readAsDataURL(input.files[0]);
        
    }
}
    