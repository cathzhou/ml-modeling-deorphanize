Programmatic data access
The full dataset that we provide for download can be found on the Downloadable data page. This section explain how to only download a subset of the data provided in XML, RDF, TSV or JSON format.

Individual entry
Data for an individual entry can be accessed in XML, RDF (trig), TSV or JSON format by adding the corresponding format extension to the ensembl gene ID as in the below URLs:

www.proteinatlas.org/ENSG00000134057.xml
www.proteinatlas.org/ENSG00000134057.trig
www.proteinatlas.org/ENSG00000134057.tsv
www.proteinatlas.org/ENSG00000134057.json
The XML format is the most comprehensive dataset of these. The TSV and JSON format contains the same data

Search queries
If you want to download a subset of the entries it is possible to do so by expressing it as a search query. To get the query use the search functionality to create the query, click search. You can then use the links at the top of search results to download the data in the specified format. The search supports the following query parameters for downloads:

Parameter	Value	Info
format	xml,trig,tsv,json	Format of the data. Required for downloads
compress	yes,no	Whether the result should be returned gz-compressed or not. The response may transparently be compressed even with compress=no if the client supports it.
The number of entries in the result set can be found in the response header X-Total-Results. It is possible to send a HEAD request to only get a count of the entries. The server may send the following status codes:

Status code	Description
400	Bad request. There was a problem in the request or the query returns too many results.
500	Server side error in the processing of the request.
Customized TSV/JSON data
In addition to the above, it is possible to customize the retrieval of the data in TSV and JSON format. This is useful if you only need certain columns of the full dataset available in either TSV or JSON format. The URL to this api is: www.proteinatlas.org/api/search_download.php

Example:

www.proteinatlas.org/api/search_download.php?search=P53&format=json&columns=g,gs&compress=no
Parameter	Value	Info
search	<Search string>	Search string for gene list. Required
format	json,tsv	Format of the data. Required
columns	<comma separated list of specifiers>	Columns for download. The specifiers for the columns can be found in the table below. Required
compress	yes,no	Whether the result should be returned gz-compressed or not. The response may transparently be compressed even with compress=no if the client supports it.
The table below shows the Column names in the data files with the corresponding name for use in the columns parameter in the API:

Column name	Columns parameter value
Gene	g
Gene synonym	gs
Ensembl	eg
Gene description	gd
Uniprot	up
Chromosome	chr
Position	chrp
Protein class	pc
Biological process	upbp
Molecular function	up_mf
Disease involvement	di
Evidence	pe
HPA evidence	evih
UniProt evidence	eviu
NeXtProt evidence	evin
RNA tissue specificity	rnats
RNA tissue distribution	rnatd
RNA tissue specificity score	rnatss
RNA tissue specific nTPM	rnatsm
RNA single cell type specificity	rnascs
RNA single cell type distribution	rnascd
RNA single cell type specificity score	rnascss
RNA single cell type specific nTPM	rnascsm