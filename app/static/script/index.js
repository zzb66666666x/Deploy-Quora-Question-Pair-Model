$(document).ready(function () {
    $('#check').click(function () {
        var q1 = $('#question1').find('.form-control').val()
        var q2 = $('#question2').find('.form-control').val()
        $.ajax({
            type: 'POST',
            url:'/predict',
            contentType: 'application/json;charset=UTF-8',
            data: JSON.stringify({
                'question1': q1,
                'question2': q2
            }),
            success: function (res) {
                // console.log(res.response)
                if (res.success){
                    alert(res.response)
                }else{
                    location.reload()
                }
            },
            error: function () {
                console.log('Error')
            }
        });
    });
});