window.onclick = function(event) {
  if (!event.target.matches(".dropbtn")) {
    var elements = document.getElementsByClassName("dropdown-content");
    var i;
    for (i = 0; i < elements.length; i++) {
      var element = elements[i];
      if (element.classList.contains("show")) {
        element.classList.remove("show");
      }
    }
  }
}

function chartTypeFunction() {
    document.getElementById("charttypes").classList.toggle("show");
}

function markSelection(id) {
    var elements = document.getElementById("charttypes").children;
	var i;

	for (i=0; i < elements.length; i++) {
		var element = elements[i];
		if (id == element.id) {
			var selectElem = document.getElementById(id);
			d3.select(selectElem).attr("class", "selected");
		} else {
			var selectElem = document.getElementById(element.id);
			d3.select(selectElem).attr("class", "dropdown");
		}
	}
}

var svg;


function twoDScatterPlot(filename, id) {

	markSelection(id);

	filename = "./static/data/" + filename;

	if (svg != undefined) {
        svg.selectAll('*').remove();
        d3.selectAll('svg').remove();
    }

    var margin = {top: 100, right: 20, bottom: 30, left: 40},
        width = 960 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

	var xValue = function(d) { return d[0];}, // data -> value
		xScale = d3.scaleLinear().range([0, width]), // value -> display
		xMap = function(d) { return xScale(xValue(d));}, // data -> display
		xAxis = d3.axisBottom(xScale);


	var yValue = function(d) { return d[1];}, // data -> value
		yScale = d3.scaleLinear().range([height, 0]), // value -> display
		yMap = function(d) { return yScale(yValue(d));}, // data -> display
		yAxis = d3.axisLeft(yScale);


	var cValue = function(d) { return d[Object.keys(d).length - 1];},
		color = d3.scaleOrdinal(d3.schemeCategory20);



	// add the graph canvas to the body of the webpage
	svg = d3.select("body").append("svg")
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
	  	.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	// add the tooltip area to the webpage
	var tooltip = d3.select("body").append("div")
		.attr("class", "tooltip")
		.style("opacity", 0);

	// load data
	d3.csv(filename, function(error, data) {

	  // change string (from CSV) into number format
	  data.forEach(function(d) {
		d[0] = +d[0];
		d[1] = +d[1];
	//    console.log(d);
	  });

	  // don't want dots overlapping axis, so add in buffer to data domain
	  xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
	  yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

	  // x-axis
	  svg.append("g")
		  .attr("class", "x axis")
		  .attr("transform", "translate(0," + height + ")")
		  .call(xAxis)
		.append("text")
		  .attr("class", "label")
		  .attr("x", width)
		  .attr("y", -6)
		  .style("text-anchor", "end")
		  .text("Calories");

	  // y-axis
	  svg.append("g")
		  .attr("class", "y axis")
		  .call(yAxis)
		.append("text")
		  .attr("class", "label")
		  .attr("transform", "rotate(-90)")
		  .attr("y", 6)
		  .attr("dy", ".71em")
		  .style("text-anchor", "end")
		  .text("Protein (g)");

	  // draw dots
	  svg.selectAll(".dot")
		  .data(data)
		.enter().append("circle")
		  .attr("class", "dot")
		  .attr("r", 3.5)
		  .attr("cx", xMap)
		  .attr("cy", yMap)
		  .style("fill", function(d) { return color(cValue(d));})
		  .on("mouseover", function(d) {
			  tooltip.transition()
				   .duration(200)
				   .style("opacity", .9);
			  tooltip.html("Cluster_id:" + d[Object.keys(d).length-1] + "<br/> (" + xValue(d)
				+ ", " + yValue(d) + ")")
				   .style("left", (d3.event.pageX + 5) + "px")
				   .style("top", (d3.event.pageY - 28) + "px");
		  })
		  .on("mouseout", function(d) {
			  tooltip.transition()
				   .duration(500)
				   .style("opacity", 0);
		  });

	  // draw legend
	  var legend = svg.selectAll(".legend")
		  .data(color.domain())
		.enter().append("g")
		  .attr("class", "legend")
		  .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

	  // draw legend colored rectangles
	  legend.append("rect")
		  .attr("x", width - 18)
		  .attr("width", 18)
		  .attr("height", 18)
		  .style("fill", color);

	  // draw legend text
	  legend.append("text")
		  .attr("x", width - 24)
		  .attr("y", 9)
		  .attr("dy", ".35em")
		  .style("text-anchor", "end")
		  .text(function(d) { return "Cluster Id: " + parseInt(d);})
	});
}

function scatterPlotMatrix(filename, id) {
    markSelection(id);

    filename = "./static/data/" + filename;

    if (svg != undefined) {
        svg.selectAll('*').remove();
        d3.selectAll('svg').remove();
    }
    var width = 960,
        size = 230,
        padding = 20;

    var x = d3.scaleLinear()
        .range([padding / 2, size - padding / 2]);

    var y = d3.scaleLinear()
        .range([size - padding / 2, padding / 2]);

    var xAxis = d3.axisBottom(x);

    var yAxis = d3.axisLeft(y);
      
    var color = d3.scaleOrdinal(d3.schemeCategory20);

    d3.csv(filename, function (error, data) {
        if (error) throw error;

        var domainByTrait = {},
            traits = d3.keys(data[0]),
            n = traits.length;

        traits.forEach(function (trait) {
            domainByTrait[trait] = d3.extent(data, function (d) {
                return d[trait];
            });
        });

        xAxis.tickSize(size * n);
        yAxis.tickSize(-size * n);

        var svg = d3.select("body").append("svg")
            .attr("width", size * n + padding)
            .attr("height", size * n + padding)
            .append("g")
            .attr("transform", "translate(" + padding + "," + padding / 2 + ")");

        svg.selectAll(".x.axis")
            .data(traits)
            .enter().append("g")
            .attr("class", "x axis")
            .attr("transform", function (d, i) {
                return "translate(" + (n - i - 1) * size + ",0)";
            })
            .each(function (d) {
                x.domain(domainByTrait[d]);
                d3.select(this).call(xAxis);
            });

        svg.selectAll(".y.axis")
            .data(traits)
            .enter().append("g")
            .attr("class", "y axis")
            .attr("transform", function (d, i) {
                return "translate(0," + i * size + ")";
            })
            .each(function (d) {
                y.domain(domainByTrait[d]);
                d3.select(this).call(yAxis);
            });

        var cell = svg.selectAll(".cell")
            .data(cross(traits, traits))
            .enter().append("g")
            .attr("class", "cell")
            .attr("transform", function (d) {
                return "translate(" + (n - d.i - 1) * size + "," + d.j * size + ")";
            })
            .each(plot);

        // Titles for the diagonal.
        cell.filter(function (d) {
            return d.i === d.j;
        }).append("text")
            .attr("x", padding)
            .attr("y", padding)
            .attr("dy", ".71em")
            .text(function (d) {
                return d.x;
            });

        function plot(p) {
            var cell = d3.select(this);

            x.domain(domainByTrait[p.x]);
            y.domain(domainByTrait[p.y]);

            cell.append("rect")
                .attr("class", "frame")
                .attr("x", padding / 2)
                .attr("y", padding / 2)
                .attr("width", size - padding)
                .attr("height", size - padding);

            cell.selectAll("circle")
                .data(data)
                .enter().append("circle")
                .attr("cx", function (d) {
                    return x(d[p.x]);
                })
                .attr("cy", function (d) {
                    return y(d[p.y]);
                })
                .attr("r", 4)
                .style("fill", function (d) {
                    return color(d[Object.keys(d).length - 1]);
                });
        }
    });
}
function cross(a, b) {
  var c = [], n = a.length, m = b.length, i, j;
  for (i = -1; ++i < n;) for (j = -1; ++j < m;) c.push({x: a[i], i: i, y: b[j], j: j});
  return c;
}


