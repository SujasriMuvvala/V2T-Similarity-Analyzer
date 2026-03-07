const analyzeBtn = document.getElementById("analyzeBtn");
, async () => {

    const videoInput = document.getElementById("videoInput");
    const reference = document.getElementById("referenceAnswer").value;

    if(videoInput.files.length === 0){
        alert("Upload a video first");
        return;
    }

    if(reference.trim() === ""){
        alert("Enter reference answer");
        return;
    }

    document.getElementById("videoText").value =
        "Processing video...";

    const formData = new FormData();

    formData.append("video", videoInput.files[0]);
    formData.append("reference", reference);

    try{

        const response = await fetch("/analyze",{

            method:"POST",
            body:formData

        });

        const data = await response.json();

        document.getElementById("videoText").value =
            data.transcript;

        document.getElementById("frameText").value =
            data.frame_description;

        document.getElementById("score-text").innerText =
            data.score + "% Similarity";

        document.getElementById("score-bar").style.width =
            data.score + "%";

    }

    catch(error){

        console.error(error);

        document.getElementById("videoText").value =
            "Error processing video";

    }

});